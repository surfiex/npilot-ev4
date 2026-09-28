"""
Lightweight standalone pure-Python implementation of crcmod.
Compatible with Python 3.12+ and AGNOS without external C compiler/pip dependencies.
"""

from collections.abc import Sequence

def _get_buffer_view(in_obj):
    if isinstance(in_obj, str):
        raise TypeError("Unicode-objects must be encoded before calculating a CRC")
    mv = memoryview(in_obj)
    if mv.ndim > 1:
        raise BufferError("Buffer must be single dimension")
    return mv

def _crc8(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFF
    for x in mv.tobytes():
        crc = table[x ^ crc]
    return crc

def _crc8r(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFF
    for x in mv.tobytes():
        crc = table[x ^ crc]
    return crc

def _crc16(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFF
    for x in mv.tobytes():
        crc = table[x ^ ((crc >> 8) & 0xFF)] ^ ((crc << 8) & 0xFF00)
    return crc

def _crc16r(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFF
    for x in mv.tobytes():
        crc = table[x ^ (crc & 0xFF)] ^ (crc >> 8)
    return crc

def _crc24(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ (crc >> 16 & 0xFF)] ^ ((crc << 8) & 0xFFFF00)
    return crc

def _crc24r(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ (crc & 0xFF)] ^ (crc >> 8)
    return crc

def _crc32(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ ((crc >> 24) & 0xFF)] ^ ((crc << 8) & 0xFFFFFF00)
    return crc

def _crc32r(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ (crc & 0xFF)] ^ (crc >> 8)
    return crc

def _crc64(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFFFFFFFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ ((crc >> 56) & 0xFF)] ^ ((crc << 8) & 0xFFFFFFFFFFFFFF00)
    return crc

def _crc64r(data, crc: int, table: Sequence[int]) -> int:
    mv = _get_buffer_view(data)
    crc = crc & 0xFFFFFFFFFFFFFFFF
    for x in mv.tobytes():
        crc = table[x ^ (crc & 0xFF)] ^ (crc >> 8)
    return crc

_sizeMap = {
    8: (_crc8, _crc8r),
    16: (_crc16, _crc16r),
    24: (_crc24, _crc24r),
    32: (_crc32, _crc32r),
    64: (_crc64, _crc64r),
}

def _verifyPoly(poly: int) -> int:
    for n in (8, 16, 24, 32, 64):
        low = 1 << n
        high = low * 2
        if low <= poly < high:
            return n
    raise ValueError("The degree of the polynomial must be 8, 16, 24, 32 or 64")

def _bitrev(x: int, n: int) -> int:
    y = 0
    for i in range(n):
        y = (y << 1) | (x & 1)
        x = x >> 1
    return y

def _bytecrc(crc: int, poly: int, n: int) -> int:
    mask = 1 << (n - 1)
    for _ in range(8):
        if crc & mask:
            crc = (crc << 1) ^ poly
        else:
            crc = crc << 1
    mask = (1 << n) - 1
    return crc & mask

def _bytecrc_r(crc: int, poly: int, n: int) -> int:
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ poly
        else:
            crc = crc >> 1
    mask = (1 << n) - 1
    return crc & mask

def _mkTable(poly: int, n: int) -> list[int]:
    mask = (1 << n) - 1
    poly = poly & mask
    return [_bytecrc(i << (n - 8), poly, n) for i in range(256)]

def _mkTable_r(poly: int, n: int) -> list[int]:
    mask = (1 << n) - 1
    poly = _bitrev(poly & mask, n)
    return [_bytecrc_r(i, poly, n) for i in range(256)]

def mkCrcFun(poly: int, initCrc: int = ~0, rev: bool = True, xorOut: int = 0):
    sizeBits = _verifyPoly(poly)
    mask = (1 << sizeBits) - 1
    initCrc = initCrc & mask
    xorOut = xorOut & mask

    if rev:
        table = _mkTable_r(poly, sizeBits)
        fun = _sizeMap[sizeBits][1]
    else:
        table = _mkTable(poly, sizeBits)
        fun = _sizeMap[sizeBits][0]

    if xorOut == 0:
        def crcfun(data, crc=initCrc):
            return fun(data, crc, table)
    else:
        def crcfun(data, crc=initCrc):
            return xorOut ^ fun(data, xorOut ^ crc, table)

    return crcfun

class Crc:
    def __init__(self, poly: int, initCrc: int = ~0, rev: bool = True, xorOut: int = 0):
        self.poly = poly
        self.initCrc = initCrc
        self.rev = rev
        self.xorOut = xorOut
        self._crcfun = mkCrcFun(poly, initCrc, rev, xorOut)
        self.crcValue = initCrc

    def update(self, data):
        self.crcValue = self._crcfun(data, self.crcValue)

    def digest(self):
        sizeBits = _verifyPoly(self.poly)
        return self.crcValue.to_bytes(sizeBits // 8, byteorder="big")

    def hexdigest(self):
        return self.digest().hex()

__all__ = ["mkCrcFun", "Crc"]
