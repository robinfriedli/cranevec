# cranevec

Vector implementation that is generic over its implementation and offers implementations that either fully use inline
storage (the capacity of which can be specified using const generics) or dynamically use the heap once it outgrows the
inline array or simply allocate to the heap like the std implementation. Additionally, the Vec provides fallible operations
to gracefully handle allocation failure.
