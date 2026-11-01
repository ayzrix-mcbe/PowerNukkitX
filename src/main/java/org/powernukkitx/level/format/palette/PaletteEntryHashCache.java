package org.powernukkitx.level.format.palette;

import io.netty.buffer.ByteBuf;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Remembers the block state hash and stored version of persistent palette entries by their exact
 * stored bytes.
 * <p>
 * Reading a palette entry from disk parses the NBT tag, rebuilds it without {@code version} and
 * serialises it again only to hash it. Both results depend on nothing but the entry's bytes, so an
 * entry seen before (stone, air, dirt...) is answered from its bytes without parsing anything. Only
 * the hash is cached; turning it into a block state still goes through the registry every time, so
 * nothing here can go stale.
 * <p>
 * The end of a tag is found with a skip-only scan of the NBT structure. An entry is remembered only
 * when that scan ends exactly where the real NBT reader stopped, so a hit always consumes the same
 * bytes and yields the same values as a full read. Anything the scan does not understand simply
 * misses and takes the normal path. Safe to use from several chunk-loading threads.
 */
final class PaletteEntryHashCache {
    /** Entries longer than this are never stored; a block state tag is normally under 200 bytes. */
    static final int MAX_ENTRY_BYTES = 512;
    /**
     * Estimated heap budget for stored entries, counting each one's bytes plus
     * {@link #ENTRY_OVERHEAD}. The accounting is reserved before inserting, so it is never exceeded
     * even with several loading threads; the real heap use follows it as closely as the overhead
     * estimate does. Once spent, further entries are read normally instead of evicting anything: a
     * world uses far fewer distinct block states than this holds (about 8,000), so it is not
     * expected to fill in practice.
     */
    static final long MAX_BYTES = 2L * 1024 * 1024;
    /**
     * Estimated heap cost of one stored entry besides its bytes: key, value, map node, table slot and
     * array header. Measured with a class histogram on a 64-bit JVM with compressed references; it is
     * roughly 30 bytes higher without them (heaps above 32 GB).
     */
    static final int ENTRY_OVERHEAD = 112;
    private static final int MAX_DEPTH = 16;
    private static final ConcurrentHashMap<Probe, Entry> CACHE = new ConcurrentHashMap<>();
    private static final AtomicLong USED_BYTES = new AtomicLong();

    record Entry(int storedVersion, int blockRuntimeId) {
    }

    private PaletteEntryHashCache() {
    }

    /**
     * Captures the stored bytes of the entry starting at the buffer's reader index.
     *
     * @return a probe for {@link #lookup} and {@link #remember}, or {@code null} if the entry's end
     *         could not be found or it is too large to cache, in which case it is read normally
     */
    static @Nullable Probe probe(ByteBuf buf) {
        final int start = buf.readerIndex();
        final int end = scanTagEnd(buf, start);
        if (end < 0 || end - start > MAX_ENTRY_BYTES) return null;
        return new Probe(buf, start, end);
    }

    /**
     * Looks up a probed entry. On a hit the reader index is moved past the entry, exactly as a full
     * read would; on a miss the buffer is left untouched.
     *
     * @return the remembered values, or {@code null} on a miss
     */
    static @Nullable Entry lookup(Probe probe, ByteBuf buf) {
        final Entry entry = CACHE.get(probe);
        if (entry != null) buf.readerIndex(probe.end);
        return entry;
    }

    /**
     * Records the values a full read produced for a probed entry, where {@code readerEnd} is where
     * the NBT reader stopped. Nothing is recorded unless the scan ended at the same place and the
     * byte budget allows it. The probe itself becomes the stored key, so nothing is copied twice.
     */
    static void remember(Probe probe, int readerEnd, int storedVersion, int blockRuntimeId) {
        if (probe.end != readerEnd) return;
        final long cost = probe.bytes.length + ENTRY_OVERHEAD;
        if (USED_BYTES.get() + cost > MAX_BYTES) return;
        if (USED_BYTES.addAndGet(cost) > MAX_BYTES) {
            USED_BYTES.addAndGet(-cost);
            return;
        }
        if (CACHE.putIfAbsent(probe, new Entry(storedVersion, blockRuntimeId)) != null) {
            USED_BYTES.addAndGet(-cost);
        }
    }

    /** Bytes currently reserved by stored entries. */
    static long usedBytes() {
        return USED_BYTES.get();
    }

    /** Number of stored entries. */
    static int size() {
        return CACHE.size();
    }

    /** Empties the cache; tests only. */
    static synchronized void clear() {
        CACHE.clear();
        USED_BYTES.set(0);
    }

    /**
     * Returns the index just past the root tag starting at {@code start}, or -1 if the bytes are
     * truncated, longer than {@link #MAX_ENTRY_BYTES}, nested deeper than the NBT reader allows, or
     * use a type it does not know. The scan never reads past that size, and positions are computed
     * in {@code long} so they cannot overflow on very large buffers.
     */
    static int scanTagEnd(ByteBuf buf, int start) {
        final int limit = (int) Math.min(buf.writerIndex(), (long) start + MAX_ENTRY_BYTES);
        if ((long) start + 3 > limit) return -1;
        final int type = buf.getUnsignedByte(start);
        if (type == 0) return -1;
        return skipPayload(buf, type, fits((long) start + 3 + buf.getUnsignedShortLE(start + 1), limit), limit, MAX_DEPTH);
    }

    private static int skipPayload(ByteBuf buf, int type, int pos, int limit, int depth) {
        if (pos < 0) return -1;
        switch (type) {
            case 1: return fits((long) pos + 1, limit);                   // byte
            case 2: return fits((long) pos + 2, limit);                   // short
            case 3: case 5: return fits((long) pos + 4, limit);           // int, float
            case 4: case 6: return fits((long) pos + 8, limit);           // long, double
            case 7: return array(buf, pos, limit, 1);                     // byte[]
            case 8:                                                       // string
                if ((long) pos + 2 > limit) return -1;
                return fits((long) pos + 2 + buf.getUnsignedShortLE(pos), limit);
            case 9: {                                                     // list
                if (depth <= 0 || (long) pos + 5 > limit) return -1;
                final int elementType = buf.getUnsignedByte(pos);
                final int count = buf.getIntLE(pos + 1);
                pos += 5;
                if (count < 0) return -1;
                if (count == 0) return pos;
                if (elementType == 0) return -1;
                for (int i = 0; i < count && pos >= 0; i++) pos = skipPayload(buf, elementType, pos, limit, depth - 1);
                return pos;
            }
            case 10: {                                                    // compound
                if (depth <= 0) return -1;
                while (true) {
                    if ((long) pos + 1 > limit) return -1;
                    final int childType = buf.getUnsignedByte(pos++);
                    if (childType == 0) return pos;
                    if ((long) pos + 2 > limit) return -1;
                    pos = fits((long) pos + 2 + buf.getUnsignedShortLE(pos), limit);
                    pos = skipPayload(buf, childType, pos, limit, depth - 1);
                    if (pos < 0) return -1;
                }
            }
            case 11: return array(buf, pos, limit, 4);                    // int[]
            case 12: return array(buf, pos, limit, 8);                    // long[]
            default: return -1;
        }
    }

    private static int array(ByteBuf buf, int pos, int limit, int elementSize) {
        if ((long) pos + 4 > limit) return -1;
        final int length = buf.getIntLE(pos);
        if (length < 0) return -1;
        return fits((long) pos + 4 + (long) length * elementSize, limit);
    }

    private static int fits(long end, int limit) {
        return end <= limit ? (int) end : -1;
    }

    /**
     * Exact stored bytes of an entry, and where it ends in the buffer it was read from. It keeps no
     * reference to that buffer, so a stored probe never holds chunk data alive. Copying the
     * bytes out once and hashing the copy measured faster than hashing and comparing them in place
     * through the buffer.
     */
    static final class Probe {
        private final byte[] bytes;
        private final int hash;
        private final int end;

        private Probe(ByteBuf buf, int start, int end) {
            this.end = end;
            this.bytes = new byte[end - start];
            buf.getBytes(start, this.bytes);
            this.hash = Arrays.hashCode(this.bytes);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Probe other && other.hash == hash && Arrays.equals(other.bytes, bytes);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
