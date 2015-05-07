from libcpp.utility cimport pair

cdef extern from "<unordered_set>" namespace "std" nogil:
    cdef cppclass unordered_set[T]:
        cppclass iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        unordered_set() nogil
        unordered_set(unordered_set&) nogil
        #unordered_set& operator=(unordered_set&)
        bint operator==(unordered_set&, unordered_set&) nogil
        bint operator!=(unordered_set&, unordered_set&) nogil
        bint operator<(unordered_set&, unordered_set&) nogil
        bint operator>(unordered_set&, unordered_set&) nogil
        bint operator<=(unordered_set&, unordered_set&) nogil
        bint operator>=(unordered_set&, unordered_set&) nogil
        iterator begin() nogil
        void clear() nogil
        size_t count(T&) nogil
        bint empty() nogil
        iterator end() nogil
        pair[iterator, iterator] equal_range(T&) nogil
        void erase(iterator) nogil
        void erase(iterator, iterator) nogil
        size_t erase(T&) nogil
        iterator find(T&) nogil
        pair[iterator, bint] insert(T&) nogil
        iterator insert(iterator, T&) nogil
        iterator lower_bound(T&) nogil
        size_t max_size() nogil
        size_t size() nogil
        void swap(unordered_set&) nogil
        iterator upper_bound(T&) nogil
