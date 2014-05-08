from libcpp.utility cimport pair
 
cdef extern from "<tr1/unordered_map>" namespace "std::tr1":
    cdef cppclass unordered_map[T, U]:
        cppclass iterator:
            pair[T, U]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        unordered_map()
        unordered_map(unordered_map&)
        U& operator[](T&) nogil
        # unordered_map& operator=(unordered_map&)
        U& at(T&) nogil
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
        pair[iterator, bint] insert(pair[T, U]) nogil
        iterator insert(iterator, pair[T, U]) nogil
        void insert(input_iterator, input_iterator)
        size_t max_size() nogil
        void rehash(size_t) nogil
        size_t size() nogil
        void swap(unordered_map&) nogil
