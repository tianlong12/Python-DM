#!/usr/bin/env python
# -*- coding: utf-8 -*
# 查找最小的k个树

def heapAdjust(A, i, length):
    pa = i
    child = 2 * i + 1
    tmp = A[i]
    while child < length:
        if child < length - 1 and A[child] < A[child + 1]:
            child += 1
        if A[pa] >= A[child]:
            break
        else:
            A[pa], A[child] = A[child], A[pa]
            pa = child
            child = 2 * pa + 1


def findKmin(A, k, length):
    for i in range(k / 2)[::-1]:
        heapAdjust(A, i, k)
    print 'The heap is :', A[:k]
    for i in xrange(k, length):
        if A[i] < A[0]:
            A[i], A[0] = A[0], A[i]
            heapAdjust(A, 0, k)
    print 'The result is :', A[:k]


if __name__ == '__main__':
    A = [6, 3, 7, 2, 9, 1, 4, 5, 11, 10, 8]
    lens = len(A)
    findKmin(A, 10, lens)