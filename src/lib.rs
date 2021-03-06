#![no_std]
#![cfg_attr(feature = "dropck_eyepatch", feature(dropck_eyepatch))]

extern crate alloc;

use alloc::{
    alloc::{alloc, dealloc, realloc, Layout},
    slice,
};

use core::fmt::{self, Debug, Display, Formatter};
use core::mem::{self, MaybeUninit};
use core::ptr::{self, swap, NonNull};
use core::{
    cmp,
    ops::{Deref, DerefMut},
};

pub type Vec<T> = CraneVec<T, DynamicContainer<T, 16>>;
pub type DynVec<T, const INLINE_SIZE: usize> = CraneVec<T, DynamicContainer<T, INLINE_SIZE>>;
pub type InlineVec<T, const SIZE: usize> = CraneVec<T, InlineContainer<T, SIZE>>;
pub type HeapVec<T> = CraneVec<T, HeapContainer<T>>;

pub struct CraneVec<T, C: Container<Element = T>> {
    container: C,
}

impl<T, const SIZE: usize> CraneVec<T, DynamicContainer<T, SIZE>> {
    pub fn new() -> Self {
        CraneVec {
            container: DynamicContainer::<T, SIZE>::new(),
        }
    }
}

impl<T, const SIZE: usize> CraneVec<T, InlineContainer<T, SIZE>> {
    pub fn new() -> Self {
        CraneVec {
            container: InlineContainer::new(),
        }
    }
}

impl<T> CraneVec<T, HeapContainer<T>> {
    pub fn new() -> Self {
        CraneVec {
            container: HeapContainer::new(),
        }
    }

    /// Deconstructs the given std Vec by leaking it and using its pointer to construct a new HeapVec,
    /// meaning this function does not have to (re-)allocate anything and simply uses the std Vec's
    /// allocated memory. This is safe since the provided Vec uses the same (Global) allocator.
    pub fn from_std_vec(mut vec: alloc::vec::Vec<T>) -> Self {
        let len = vec.len();
        let capacity = vec.capacity();
        let ptr = if vec.is_empty() {
            NonNull::dangling()
        } else {
            // SAFETY: A non-empty Vec holds a valid pointer
            unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) }
        };
        vec.leak();
        CraneVec {
            container: HeapContainer { ptr, capacity, len },
        }
    }
}

impl<T, C: Container<Element = T>> Deref for CraneVec<T, C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.container
    }
}

impl<T, C: Container<Element = T>> DerefMut for CraneVec<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.container
    }
}

impl<T, const SIZE: usize> Default for CraneVec<T, DynamicContainer<T, SIZE>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq, C: Container<Element = T>> PartialEq for CraneVec<T, C> {
    fn eq(&self, other: &Self) -> bool {
        *self.container == *other.container
    }
}

impl<T: Debug, C: Container<Element = T>> Debug for CraneVec<T, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.container, f)
    }
}

pub trait Container: DerefMut<Target = [Self::Element]> {
    type Element;

    fn push(&mut self, el: Self::Element) {
        self.try_push(el).expect("Could not allocate element due to container capacity constraints or allocation failure");
    }

    fn try_push(&mut self, el: Self::Element) -> InsertionResult<Self::Element>;

    fn read_at(&self, pos: usize) -> Option<&Self::Element>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;

    fn capacity(&self) -> usize;

    fn as_ptr(&self) -> *const Self::Element;

    fn as_mut_ptr(&mut self) -> *mut Self::Element;

    /// Reserves the minimum capacity required for `additional` more elements, favouring the nearest multiple of 2.
    ///
    /// Does nothing if the current capacity already suffices.
    ///
    /// Unlike [`Container::try_reserve_additional`] this adds `additional` to the current length,
    /// not the capacity.
    ///
    /// Returns an [`AllocationError`] if the required `additional` capacity could not be allocated.
    fn try_reserve(&mut self, additional: usize) -> AllocationResult;

    /// Reserves the minimum capacity required for `additional` more elements, favouring the nearest multiple of 2.
    ///
    /// Does nothing if the current capacity already suffices.
    ///
    /// Unlike [`Container::try_reserve_additional`] this adds `additional` to the current length,
    /// not the capacity.
    ///
    /// Panics if the required `additional` capacity could not be allocated.
    fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional)
            .expect("Could not allocate additional capacity");
    }

    /// Reserves the exact capacity required for `additional` more elements.
    ///
    /// Does nothing if the current capacity already suffices.
    ///
    /// Unlike [`Container::try_reserve_additional`] this adds `additional` to the current length,
    /// not the capacity.
    ///
    /// Returns an [`AllocationError`] if the required `additional` capacity could not be allocated.
    fn try_reserve_exact(&mut self, additional: usize) -> AllocationResult;

    /// Reserves the exact capacity required for `additional` more elements.
    ///
    /// Does nothing if the current capacity already suffices.
    ///
    /// Unlike [`Container::try_reserve_additional`] this adds `additional` to the current length,
    /// not the capacity.
    ///
    /// Panics if the required `additional` capacity could not be allocated.
    fn reserve_exact(&mut self, additional: usize) {
        self.try_reserve_exact(additional)
            .expect("Could not allocate additional capacity")
    }

    /// Increases the capacity by at least `additional_capacity`, favouring the nearest multiple of 2.
    ///
    /// Unlike [`Container::try_reserve`] this adds the `additional_capacity` to the current capacity,
    /// not the length.
    ///
    /// Returns an [`AllocationError`] if the required `additional_capacity` could not be allocated.
    fn try_reserve_additional(&mut self, additional_capacity: usize) -> AllocationResult;

    /// Increases the capacity by at least `additional_capacity`, favouring the nearest multiple of 2.
    ///
    /// Unlike [`Container::try_reserve`] this adds the `additional_capacity` to the current capacity,
    /// not the length.
    ///
    /// Panics if the required `additional_capacity` could not be allocated.
    fn reserve_additional(&mut self, additional_capacity: usize) {
        self.try_reserve_additional(additional_capacity)
            .expect("Could not allocate additional capacity")
    }

    /// Increases the capacity by exactly `additional_capacity`.
    ///
    /// Returns an [`AllocationError`] if the required `additional_capacity` could not be allocated.
    fn try_reserve_additional_exact(&mut self, additional_capacity: usize) -> AllocationResult;

    /// Increases the capacity by exactly `additional_capacity`.
    ///
    /// Panics if the required `additional_capacity` could not be allocated.
    fn reserve_additional_exact(&mut self, additional_capacity: usize) {
        self.try_reserve_additional_exact(additional_capacity)
            .expect("Could not allocate additional capacity")
    }

    /// Update the length of this vector when changing its length outside of safe functions provided by this module.
    /// Also used internally to update the length agnostic to the [`Container`] implementation.
    ///
    /// # Safety
    /// Length must be <= capacity and must match the length of initialised elements [0..len]. When reducing
    /// the length of the vector the caller needs to handle dropping elements [new_len..old_len] since the Drop implementations
    /// for Containers generally use the length to subslice and drop initialised elements.
    unsafe fn set_len(&mut self, len: usize);

    /// Remove the element at the given index by copying all elements at [index + 1..len] to index, essentially
    /// moving all elements after the provided index to the left by one. This has a worst case performance of
    /// O(n) (linear time) when removing an item from the start of the vec but preserves the order of all other elements,
    /// for an O(1) (constant time) implementation that does not preserve the order of elements see [`Container::try_swap_remove`].
    ///
    /// Returns the removed item or `None` if the index is out of bounds for [0..len].
    ///
    /// ```rust
    /// use cranevec::HeapVec;
    /// use crate::cranevec::Container;
    /// let mut vec = HeapVec::from_std_vec(vec![1, 0, 9]);
    /// assert_eq!(vec.try_remove(1), Some(0));
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec[0], 1);
    /// assert_eq!(vec[1], 9);
    /// assert_eq!(vec.try_remove(2), None);
    ///
    /// let mut alphabet = HeapVec::from_std_vec(vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']);
    /// assert_eq!(alphabet.try_remove(3), Some('d'));
    /// assert_eq!(alphabet, HeapVec::from_std_vec(vec!['a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j']));
    /// ```
    fn try_remove(&mut self, index: usize) -> Option<Self::Element> {
        let len = self.len();
        if index >= len {
            None
        } else {
            unsafe {
                let ptr = self.as_mut_ptr().add(index);
                let removed_el = ptr::read(ptr);
                ptr::copy(ptr.offset(1), ptr, len - index - 1);
                self.set_len(len - 1);
                Some(removed_el)
            }
        }
    }

    /// Remove the element at the given index by copying all elements at [index + 1..len] to index, essentially
    /// moving all elements after the provided index to the left by one. This has a worst case performance of
    /// O(n) (linear time) when removing an item from the start of the vec but preserves the order of all other elements,
    /// for an O(1) (constant time) implementation that does not preserve the order of elements see [`Container::try_swap_remove`].
    ///
    /// Returns the removed item or panics if the index is out of bounds for [0..len].
    fn remove(&mut self, index: usize) -> Self::Element {
        let result = self.try_remove(index);
        assert!(
            result.is_some(),
            "Could not remove index {} for cranevec of length {}",
            index,
            self.len()
        );
        result.unwrap()
    }

    /// Remove the element at the given index by replacing it with the last item in the vector and reducing its length.
    /// This always has a complexity of O(1) (constant time) but does not preserve the order of elements. For an
    /// implementation that is O(n) (linear time) but preserves the order of elements see [`Container::try_remove`].
    ///
    /// Returns the removed item or `None` if the index is out of bounds for [0..len].
    ///
    /// ```rust
    /// use cranevec::HeapVec;
    /// use crate::cranevec::Container;
    /// let mut vec = HeapVec::from_std_vec(vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']);
    /// assert_eq!(vec.try_swap_remove(3), Some('d'));
    /// assert_eq!(vec, HeapVec::from_std_vec(vec!['a', 'b', 'c', 'j', 'e', 'f', 'g', 'h', 'i']));
    /// ```
    fn try_swap_remove(&mut self, index: usize) -> Option<Self::Element> {
        let len = self.len();
        if index >= len {
            None
        } else {
            unsafe {
                let removed_el = ptr::read(self.as_ptr().add(index));
                let data_ptr = self.as_mut_ptr();
                ptr::copy(data_ptr.add(len - 1), data_ptr.add(index), 1);
                self.set_len(len - 1);
                Some(removed_el)
            }
        }
    }

    /// Remove the element at the given index by replacing it with the last item in the vector and reducing its length.
    /// This always has a complexity of O(1) (constant time) but does not preserve the order of elements. For an
    /// implementation that is O(n) (linear time) but preserves the order of elements see [`Container::try_remove`].
    ///
    /// Returns the removed item or panics if the index is out of bounds for [0..len].
    fn swap_remove(&mut self, index: usize) -> Self::Element {
        let result = self.try_swap_remove(index);
        assert!(
            result.is_some(),
            "Could not remove index {} for cranevec of length {}",
            index,
            self.len()
        );
        result.unwrap()
    }
}

pub struct InlineContainer<T, const SIZE: usize> {
    data: [MaybeUninit<T>; SIZE],
    len: usize,
    #[cfg(feature = "dropck_eyepatch")]
    _marker: core::marker::PhantomData<T>,
}

impl<T, const SIZE: usize> Container for InlineContainer<T, SIZE> {
    type Element = T;

    fn try_push(&mut self, el: T) -> InsertionResult<T> {
        if self.len < self.capacity() {
            unsafe {
                self.data.get_unchecked_mut(self.len).as_mut_ptr().write(el);
            }
            self.len += 1;
            Ok(())
        } else {
            Err(InsertionError::AllocationError(
                el,
                AllocationError::CapacityExceeded,
            ))
        }
    }

    fn read_at(&self, pos: usize) -> Option<&T> {
        if pos < self.len {
            unsafe { Some(&*(self.data.get_unchecked(pos) as *const MaybeUninit<T> as *const T)) }
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            SIZE
        }
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    fn try_reserve(&mut self, additional: usize) -> AllocationResult {
        let required_capacity = self
            .len
            .checked_add(additional)
            .ok_or(AllocationError::ArithmeticOverflow)?;

        if required_capacity > SIZE {
            Err(AllocationError::CapacityExceeded)
        } else {
            Ok(())
        }
    }

    fn try_reserve_exact(&mut self, additional: usize) -> AllocationResult {
        let required_capacity = self
            .len
            .checked_add(additional)
            .ok_or(AllocationError::ArithmeticOverflow)?;

        if required_capacity > SIZE {
            Err(AllocationError::CapacityExceeded)
        } else {
            Ok(())
        }
    }

    fn try_reserve_additional(&mut self, additional_capacity: usize) -> AllocationResult {
        if additional_capacity > 0 {
            Err(AllocationError::CapacityExceeded)
        } else {
            Ok(())
        }
    }

    fn try_reserve_additional_exact(&mut self, additional_capacity: usize) -> AllocationResult {
        if additional_capacity > 0 {
            Err(AllocationError::CapacityExceeded)
        } else {
            Ok(())
        }
    }

    unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<T, const SIZE: usize> Drop for InlineContainer<T, SIZE> {
    fn drop(&mut self) {
        unsafe {
            let init = &mut *(slice::from_raw_parts_mut(self.data.as_mut_ptr(), self.len)
                as *mut [MaybeUninit<T>] as *mut [T]);
            ptr::drop_in_place(init);
        }
    }
}

#[cfg(feature = "dropck_eyepatch")]
unsafe impl<#[may_dangle] T, const SIZE: usize> Drop for InlineContainer<T, SIZE> {
    fn drop(&mut self) {
        unsafe {
            let init = &mut *(slice::from_raw_parts_mut(self.data.as_mut_ptr(), self.len)
                as *mut [MaybeUninit<T>] as *mut [T]);
            ptr::drop_in_place(init);
        }
    }
}

impl<T, const SIZE: usize> Deref for InlineContainer<T, SIZE> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl<T, const SIZE: usize> DerefMut for InlineContainer<T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

impl<T, const SIZE: usize> Default for InlineContainer<T, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const SIZE: usize> InlineContainer<T, SIZE> {
    pub fn new() -> Self {
        Self {
            // SAFETY: An uninitialized `[MaybeUninit<_>; SIZE]` is valid.
            // See (currently nightly) only [`uninit_array`](https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.uninit_array)
            data: unsafe { MaybeUninit::<[MaybeUninit<T>; SIZE]>::uninit().assume_init() },
            len: 0,
            #[cfg(feature = "dropck_eyepatch")]
            _marker: core::marker::PhantomData,
        }
    }
}

pub struct HeapContainer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    len: usize,
    #[cfg(feature = "dropck_eyepatch")]
    _marker: core::marker::PhantomData<T>,
}

impl<T> Container for HeapContainer<T> {
    type Element = T;

    fn try_push(&mut self, el: T) -> InsertionResult<T> {
        let curr_len = self.len;
        if curr_len == self.capacity() {
            if let Err(e) = self.try_reserve_additional(1) {
                return Err(InsertionError::AllocationError(el, e));
            }
        }

        // SAFETY:
        // Length is always lower than the capacity, thus the pointer is in range of the allocated object.
        unsafe { self.ptr.as_ptr().add(curr_len).write(el) };

        self.len += 1;
        Ok(())
    }

    fn read_at(&self, pos: usize) -> Option<&T> {
        if pos < self.len {
            unsafe { Some(&*self.ptr.as_ptr().add(pos)) }
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.capacity
        }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn try_reserve(&mut self, additional: usize) -> AllocationResult {
        let curr_capacity = self.capacity;
        let required_capacity = self
            .len
            .checked_add(additional)
            .ok_or(AllocationError::ArithmeticOverflow)?;
        let new_capacity = cmp::max(
            cmp::max(curr_capacity * 2, required_capacity),
            Self::MIN_CAP,
        );

        if required_capacity > curr_capacity {
            self.try_reserve_capacity(new_capacity, required_capacity)
        } else {
            Ok(())
        }
    }

    fn try_reserve_exact(&mut self, additional: usize) -> AllocationResult {
        let curr_capacity = self.capacity;
        let required_capacity = self
            .len
            .checked_add(additional)
            .ok_or(AllocationError::ArithmeticOverflow)?;

        if required_capacity > curr_capacity {
            self.try_reserve_capacity(required_capacity, required_capacity)
        } else {
            Ok(())
        }
    }

    fn try_reserve_additional(&mut self, additional_capacity: usize) -> AllocationResult {
        let curr_capacity = self.capacity;
        let required_capacity = curr_capacity
            .checked_add(additional_capacity)
            .ok_or(AllocationError::ArithmeticOverflow)?;
        let new_capacity = cmp::max(
            cmp::max(curr_capacity * 2, required_capacity),
            Self::MIN_CAP,
        );

        self.try_reserve_capacity(new_capacity, required_capacity)
    }

    fn try_reserve_additional_exact(&mut self, additional_capacity: usize) -> AllocationResult {
        let curr_capacity = self.capacity;
        let required_capacity = curr_capacity
            .checked_add(additional_capacity)
            .ok_or(AllocationError::ArithmeticOverflow)?;

        self.try_reserve_capacity(required_capacity, required_capacity)
    }

    unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }
}

#[cfg(not(feature = "dropck_eyepatch"))]
impl<T> Drop for HeapContainer<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            if let Some(layout) = self.current_layout() {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

#[cfg(feature = "dropck_eyepatch")]
unsafe impl<#[may_dangle] T> Drop for HeapContainer<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            if let Some(layout) = self.current_layout() {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for HeapContainer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for HeapContainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

impl<T> Default for HeapContainer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HeapContainer<T> {
    pub fn new() -> Self {
        HeapContainer {
            ptr: NonNull::dangling(),
            capacity: 0,
            len: 0,
            #[cfg(feature = "dropck_eyepatch")]
            _marker: core::marker::PhantomData,
        }
    }

    fn new_with_len(len: usize) -> Self {
        HeapContainer {
            ptr: NonNull::dangling(),
            capacity: 0,
            len,
            #[cfg(feature = "dropck_eyepatch")]
            _marker: core::marker::PhantomData,
        }
    }

    // From std Vec:
    // Skip to:
    // - 8 if the element size is 1, because any heap allocators is likely
    //   to round up a request of less than 8 bytes to at least 8 bytes.
    // - 4 if elements are moderate-sized (<= 1 KiB).
    // - 1 otherwise, to avoid wasting too much space for very short Vecs.
    const MIN_CAP: usize = if mem::size_of::<T>() == 1 {
        8
    } else if mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    };

    const MAX_SLICE_BYTE_LEN: usize = isize::MAX as usize;

    fn try_reserve_capacity(
        &mut self,
        new_capacity: usize,
        required_capacity: usize,
    ) -> AllocationResult {
        // SAFETY:
        // Layout is current array layout and allocated / reallocated size is multiple of size of T and <= isize::MAX
        let ptr = unsafe {
            match self.current_layout() {
                Some(layout) => {
                    let new_size = new_capacity
                        .checked_mul(mem::size_of::<T>())
                        .ok_or(AllocationError::ArithmeticOverflow)?;

                    if new_size > Self::MAX_SLICE_BYTE_LEN && new_capacity > required_capacity {
                        let required_size = required_capacity
                            .checked_mul(mem::size_of::<T>())
                            .ok_or(AllocationError::ArithmeticOverflow)?;
                        if Self::MAX_SLICE_BYTE_LEN >= required_size {
                            realloc(
                                self.ptr.as_ptr() as *mut u8,
                                layout,
                                Self::MAX_SLICE_BYTE_LEN,
                            )
                        } else {
                            return Err(AllocationError::ArithmeticOverflow);
                        }
                    } else {
                        realloc(self.ptr.as_ptr() as *mut u8, layout, new_size)
                    }
                }
                None => {
                    let layout = Layout::array::<T>(new_capacity)
                        .map_err(|_| AllocationError::ArithmeticOverflow)?;

                    if layout.size() > Self::MAX_SLICE_BYTE_LEN {
                        if Self::MAX_SLICE_BYTE_LEN >= required_capacity {
                            let layout = Layout::array::<T>(Self::MAX_SLICE_BYTE_LEN)
                                .map_err(|_| AllocationError::ArithmeticOverflow)?;
                            alloc(layout)
                        } else {
                            return Err(AllocationError::ArithmeticOverflow);
                        }
                    } else {
                        alloc(layout)
                    }
                }
            }
        };
        self.ptr = NonNull::new(ptr as *mut T).ok_or(AllocationError::OutOfMemory)?;
        self.capacity = new_capacity;
        Ok(())
    }

    #[inline]
    fn current_layout(&self) -> Option<Layout> {
        if mem::size_of::<T>() == 0 || self.capacity == 0 {
            None
        } else {
            // SAFETY: should not overflow because if it would, we could not have allocated it
            unsafe {
                Some(Layout::from_size_align_unchecked(
                    mem::size_of::<T>() * self.capacity,
                    mem::align_of::<T>(),
                ))
            }
        }
    }
}

pub enum DynamicData<T, const INLINE_SIZE: usize> {
    Inline(InlineContainer<T, INLINE_SIZE>),
    Heap(HeapContainer<T>),
}

pub struct DynamicContainer<T, const INLINE_SIZE: usize> {
    data: DynamicData<T, INLINE_SIZE>,
}

impl<T, const INLINE_SIZE: usize> DynamicContainer<T, INLINE_SIZE> {
    pub fn try_move_to_heap(&mut self) -> AllocationResult {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                Self::move_inline_to_heap(container)?;
                Ok(())
            }
            DynamicData::Heap(_) => Ok(()),
        }
    }

    fn move_inline_to_heap(
        container: &mut InlineContainer<T, INLINE_SIZE>,
    ) -> Result<HeapContainer<T>, AllocationError> {
        // pushing to inline container only fails when len reaches capacity
        let mut heap_container = HeapContainer::new_with_len(INLINE_SIZE);

        // make sure the data is reallocated to the heap and increase the capacity by at least 1
        heap_container.try_reserve_additional(cmp::max(INLINE_SIZE * 2, INLINE_SIZE + 1))?;
        unsafe { Self::move_inline_to_reserved_heap(container, heap_container) }
    }

    // SAFETY: caller must make sure the HeapContainer has allocated the required capacity to store INLINE_SIZE elements.
    unsafe fn move_inline_to_reserved_heap(
        container: &mut InlineContainer<T, INLINE_SIZE>,
        heap_container: HeapContainer<T>,
    ) -> Result<HeapContainer<T>, AllocationError> {
        // Get a pointer to the slice of the heap allocation the array should be swapped into.
        let heap_slice = ptr::slice_from_raw_parts_mut(heap_container.ptr.as_ptr(), INLINE_SIZE)
            as *mut [T; INLINE_SIZE];
        // Can safely convert [MaybeUninit<T>] to [T; INLINE_SIZE] because the array has been fully initialised.
        let inline_slice =
            container.data.as_mut() as *mut [MaybeUninit<T>] as *mut [T; INLINE_SIZE];
        // SAFETY:
        // The array is fully initialised, otherwise pushing another element would not have failed, thus assuming all elements
        // in the array to be valid and swapping them into the heap allocation is safe. The uninitialised memory of the heap
        // allocation is swapped into the array, which is safe because the array is of type [MaybeUninit<T>], thus does not
        // require its data to be initialised.
        swap(heap_slice, inline_slice);
        // make sure that the inline storage is marked to be empty, now that the array has been replaced with uninitialised memory
        container.len = 0;
        Ok(heap_container)
    }
}

impl<T, const INLINE_SIZE: usize> Container for DynamicContainer<T, INLINE_SIZE> {
    type Element = T;

    fn try_push(&mut self, el: T) -> InsertionResult<T> {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                if let Err(e) = container.try_push(el) {
                    let el = e.into_inner();
                    match Self::move_inline_to_heap(container) {
                        Ok(mut heap_container) => {
                            // Since the memory has already been reserved, this operation should always succeed.
                            heap_container.push(el);
                            self.data = DynamicData::Heap(heap_container);
                        }
                        Err(allocation_error) => {
                            return Err(InsertionError::AllocationError(el, allocation_error))
                        }
                    };
                }
            }
            DynamicData::Heap(ref mut container) => {
                container.try_push(el)?;
            }
        }

        Ok(())
    }

    fn read_at(&self, pos: usize) -> Option<&T> {
        match self.data {
            DynamicData::Inline(ref container) => container.read_at(pos),
            DynamicData::Heap(ref container) => container.read_at(pos),
        }
    }

    fn len(&self) -> usize {
        match self.data {
            DynamicData::Inline(ref container) => container.len,
            DynamicData::Heap(ref container) => container.len,
        }
    }

    fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            match self.data {
                DynamicData::Inline(ref container) => container.capacity(),
                DynamicData::Heap(ref container) => container.capacity(),
            }
        }
    }

    fn as_ptr(&self) -> *const T {
        match self.data {
            DynamicData::Inline(ref container) => container.as_ptr(),
            DynamicData::Heap(ref container) => container.as_ptr(),
        }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        match self.data {
            DynamicData::Inline(ref mut container) => container.as_mut_ptr(),
            DynamicData::Heap(ref mut container) => container.as_mut_ptr(),
        }
    }

    fn try_reserve(&mut self, additional: usize) -> AllocationResult {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                let required_capacity = container
                    .len
                    .checked_add(additional)
                    .ok_or(AllocationError::ArithmeticOverflow)?;

                if required_capacity > INLINE_SIZE {
                    let mut heap_container = HeapContainer::new_with_len(container.len);

                    heap_container.try_reserve(additional)?;
                    let heap_container =
                        unsafe { Self::move_inline_to_reserved_heap(container, heap_container) }?;
                    self.data = DynamicData::Heap(heap_container);
                    Ok(())
                } else {
                    Ok(())
                }
            }
            DynamicData::Heap(ref mut container) => container.try_reserve(additional),
        }
    }

    fn try_reserve_exact(&mut self, additional: usize) -> AllocationResult {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                let required_capacity = container
                    .len
                    .checked_add(additional)
                    .ok_or(AllocationError::ArithmeticOverflow)?;

                if required_capacity > INLINE_SIZE {
                    let mut heap_container = HeapContainer::new_with_len(container.len);

                    heap_container.try_reserve_exact(additional)?;
                    let heap_container =
                        unsafe { Self::move_inline_to_reserved_heap(container, heap_container) }?;
                    self.data = DynamicData::Heap(heap_container);
                    Ok(())
                } else {
                    Ok(())
                }
            }
            DynamicData::Heap(ref mut container) => container.try_reserve_exact(additional),
        }
    }

    fn try_reserve_additional(&mut self, additional_capacity: usize) -> AllocationResult {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                let mut heap_container = HeapContainer::new_with_len(container.len);

                heap_container.try_reserve_additional(INLINE_SIZE + additional_capacity)?;
                let heap_container =
                    unsafe { Self::move_inline_to_reserved_heap(container, heap_container) }?;
                self.data = DynamicData::Heap(heap_container);
                Ok(())
            }
            DynamicData::Heap(ref mut container) => {
                container.try_reserve_additional(additional_capacity)
            }
        }
    }

    fn try_reserve_additional_exact(&mut self, additional_capacity: usize) -> AllocationResult {
        match self.data {
            DynamicData::Inline(ref mut container) => {
                let mut heap_container = HeapContainer::new_with_len(container.len);

                heap_container.try_reserve_additional_exact(INLINE_SIZE + additional_capacity)?;
                let heap_container =
                    unsafe { Self::move_inline_to_reserved_heap(container, heap_container) }?;
                self.data = DynamicData::Heap(heap_container);
                Ok(())
            }
            DynamicData::Heap(ref mut container) => {
                container.try_reserve_additional_exact(additional_capacity)
            }
        }
    }

    unsafe fn set_len(&mut self, len: usize) {
        match self.data {
            DynamicData::Inline(ref mut container) => container.len = len,
            DynamicData::Heap(ref mut container) => container.len = len,
        }
    }
}

impl<T, const INLINE_SIZE: usize> Deref for DynamicContainer<T, INLINE_SIZE> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, const INLINE_SIZE: usize> DerefMut for DynamicContainer<T, INLINE_SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl<T, const INLINE_SIZE: usize> Default for DynamicContainer<T, INLINE_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const INLINE_SIZE: usize> DynamicContainer<T, INLINE_SIZE> {
    pub fn new() -> Self {
        if INLINE_SIZE > 0 {
            DynamicContainer {
                data: DynamicData::Inline(InlineContainer::new()),
            }
        } else {
            DynamicContainer {
                data: DynamicData::Heap(HeapContainer::new()),
            }
        }
    }
}

pub enum InsertionError<T> {
    AllocationError(T, AllocationError),
}

impl<T> InsertionError<T> {
    pub fn into_inner(self) -> T {
        match self {
            InsertionError::AllocationError(elem, ..) => elem,
        }
    }
}

impl<T> Debug for InsertionError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            InsertionError::AllocationError(.., ref e) => {
                write!(f, "InsertionError::AllocationError: {}", e)
            }
        }
    }
}

impl<T> Display for InsertionError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <InsertionError<T> as Debug>::fmt(self, f)
    }
}

#[derive(Debug)]
pub enum AllocationError {
    /// The capacity of the container has been exceeded and the container does not support resizing.
    CapacityExceeded,
    /// The heap allocator could not request more memory.
    OutOfMemory,
    /// Calculating byte offset resulted in arithmetic overflow, generally byte offset cannot be larger than `isize::MAX`.
    ArithmeticOverflow,
}

impl Display for AllocationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AllocationError::CapacityExceeded => write!(f, "CapacityExceeded"),
            AllocationError::OutOfMemory => write!(f, "OutOfMemory"),
            AllocationError::ArithmeticOverflow => write!(f, "ArithmeticOverflow"),
        }
    }
}

pub type AllocationResult = Result<(), AllocationError>;
pub type InsertionResult<T> = Result<(), InsertionError<T>>;

#[cfg(test)]
mod tests {

    extern crate std;

    use alloc::vec;

    use crate::{Container, CraneVec, DynVec, HeapVec, InlineContainer, InlineVec, Vec};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct PrintOnDrop {
        val: usize,
    }
    impl PrintOnDrop {
        fn new() -> Self {
            Self { val: 7 }
        }
    }
    impl Drop for PrintOnDrop {
        fn drop(&mut self) {
            std::eprintln!("Dropped {}", self.val);
        }
    }

    struct IncrementOnDrop<'a> {
        counter: &'a AtomicUsize,
    }

    impl<'a> IncrementOnDrop<'a> {
        fn new(counter: &'a AtomicUsize) -> Self {
            Self { counter }
        }
    }

    impl<'a> Drop for IncrementOnDrop<'a> {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn it_works() {
        let mut vec = CraneVec::<IncrementOnDrop, InlineContainer<IncrementOnDrop, 16>>::new();
        let counter = AtomicUsize::new(0);
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        assert_eq!(vec.len(), 5);
        drop(vec);
        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[ignore]
    #[test]
    fn it_works_bench() {
        let instant = std::time::Instant::now();
        let mut stdvec = std::vec::Vec::new();
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        std::eprintln!("std vec nanos {}", instant.elapsed().as_nanos());
        drop(stdvec);

        let instant = std::time::Instant::now();
        let mut vec = CraneVec::<PrintOnDrop, InlineContainer<PrintOnDrop, 16>>::new();
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        std::eprintln!("cranevec nanos {}", instant.elapsed().as_nanos());
        drop(vec);

        InlineVec::<PrintOnDrop, 16>::new();
    }

    #[test]
    fn test_heap() {
        let mut vec = HeapVec::new();
        let counter = AtomicUsize::new(0);
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        vec.push(IncrementOnDrop::new(&counter));
        assert_eq!(vec.len(), 5);
        drop(vec);
        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[ignore]
    #[test]
    fn test_heap_bench() {
        let instant = std::time::Instant::now();
        let mut stdvec = std::vec::Vec::new();
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        stdvec.push(PrintOnDrop::new());
        std::eprintln!("std vec nanos {}", instant.elapsed().as_nanos());
        drop(stdvec);

        let instant = std::time::Instant::now();
        let mut vec = HeapVec::new();
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        vec.push(PrintOnDrop::new());
        std::eprintln!("cranevec nanos {}", instant.elapsed().as_nanos());
        drop(vec);

        HeapVec::<PrintOnDrop>::new();
    }

    #[test]
    fn test_heap_large() {
        let mut vec = HeapVec::new();
        let counter = AtomicUsize::new(0);
        for _ in 0..500000 {
            vec.push(IncrementOnDrop::new(&counter));
        }
        assert_eq!(vec.len(), 500000);
        drop(vec);
        assert_eq!(counter.load(Ordering::Relaxed), 500000);
    }

    #[ignore]
    #[test]
    fn test_heap_large_bench() {
        let instant = std::time::Instant::now();
        let mut stdvec = std::vec::Vec::new();
        for _ in 0..500000 {
            stdvec.push(PrintOnDrop::new());
        }
        std::eprintln!("std vec nanos {}", instant.elapsed().as_nanos());
        drop(stdvec);

        let instant = std::time::Instant::now();
        let mut vec = HeapVec::new();
        for _ in 0..500000 {
            vec.push(PrintOnDrop::new());
        }
        std::eprintln!("cranevec nanos {}", instant.elapsed().as_nanos());
        drop(vec);

        HeapVec::<PrintOnDrop>::new();
    }

    #[test]
    fn test_dynamic() {
        let mut vec = DynVec::<IncrementOnDrop, 16>::new();
        let counter = AtomicUsize::new(0);
        for _ in 0..20 {
            vec.push(IncrementOnDrop::new(&counter));
        }
        assert_eq!(vec.len(), 20);
        drop(vec);
        assert_eq!(counter.load(Ordering::Relaxed), 20);
    }

    #[ignore]
    #[test]
    fn test_dynamic_bench() {
        let instant = std::time::Instant::now();
        let mut stdvec = std::vec::Vec::new();
        for _ in 0..20 {
            stdvec.push(PrintOnDrop::new());
        }
        std::eprintln!("std vec nanos {}", instant.elapsed().as_nanos());
        drop(stdvec);

        let instant = std::time::Instant::now();
        let mut vec = Vec::new();
        for _ in 0..20 {
            vec.push(PrintOnDrop::new());
        }
        std::eprintln!("cranevec nanos {}", instant.elapsed().as_nanos());
        drop(vec);

        HeapVec::<PrintOnDrop>::new();
    }

    #[test]
    fn test_get() {
        let mut vec = DynVec::<usize, 16>::new();

        for i in 0..15 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 15);
        assert_eq!(vec.capacity(), 16);

        for i in 0..20 {
            if i < 15 {
                assert_eq!(vec.get(i), Some(&i));
            } else {
                assert_eq!(vec.get(i), None);
            }
        }

        for i in 15..=250 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 251);
        assert_eq!(vec.capacity(), 256);

        for i in 15..300 {
            if i <= 250 {
                assert_eq!(vec.get(i), Some(&i));
            } else {
                assert_eq!(vec.get(i), None);
            }
        }
    }

    #[test]
    fn test_iter() {
        let mut vec = Vec::new();

        for i in 0..10000 {
            vec.push(i);
        }

        let mut idx = 0;
        for i in vec.iter() {
            assert_eq!(*i, idx);
            idx += 1;
        }
    }

    #[should_panic(
        expected = "Could not allocate element due to container capacity constraints or allocation failure: InsertionError::AllocationError: CapacityExceeded"
    )]
    #[test]
    fn test_inline_overflow() {
        let mut vec = InlineVec::<i32, 4>::new();

        for i in 0..5 {
            vec.push(i);
        }
    }

    #[test]
    fn test_reserve() {
        let mut vec = HeapVec::<i32>::new();

        vec.reserve(16);
        assert_eq!(vec.capacity(), 16);

        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        assert_eq!(vec.capacity(), 16);
        vec.reserve_exact(10);
        assert_eq!(vec.capacity(), 16);

        vec.reserve_additional(10);
        assert_eq!(vec.capacity(), 32);

        vec.reserve_additional_exact(5);
        assert_eq!(vec.capacity(), 37);

        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        assert_eq!(vec.capacity(), 37);

        assert_eq!(vec.iter().sum::<i32>(), 10);
    }

    #[test]
    fn test_reserve_dynamic() {
        let mut vec = DynVec::<i32, 16>::new();

        assert_eq!(vec.capacity(), 16);
        vec.reserve(16);
        assert_eq!(vec.capacity(), 16);

        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);

        assert_eq!(vec.capacity(), 16);
        assert_eq!(vec.len(), 5);
        vec.reserve(10);
        assert_eq!(vec.capacity(), 16);
        assert_eq!(vec.len(), 5);

        vec.reserve(16);
        assert_eq!(vec.capacity(), 21);
        assert_eq!(vec.len(), 5);

        for i in vec.iter() {
            assert_eq!(*i, 1);
        }
        assert_eq!(vec.iter().sum::<i32>(), 5);

        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);
        vec.push(1);

        assert_eq!(vec.capacity(), 21);
        assert_eq!(vec.len(), 10);

        for i in vec.iter() {
            assert_eq!(*i, 1);
        }
        assert_eq!(vec.iter().sum::<i32>(), 10);
    }

    #[test]
    fn test_partial_eq() {
        let mut vec1 = Vec::new();
        let mut vec2 = Vec::new();

        vec1.push(1);
        vec1.push(0);
        vec1.push(9);
        vec2.push(1);
        vec2.push(0);
        vec2.push(9);

        assert_eq!(vec1, vec2);

        let mut vec3 = Vec::new();
        let mut vec4 = Vec::new();
        let mut vec5 = Vec::new();

        vec3.push(4);
        vec3.push(5);
        vec4.push(4);
        vec4.push(6);
        vec5.push(4);

        assert!(vec3 != vec4);
        assert!(vec3 != vec4);
        assert!(vec4 != vec5);
    }

    #[test]
    fn test_from_std_vec() {
        let src = vec![1, 0, 9];
        let capacity = src.capacity();
        let mut vec = HeapVec::from_std_vec(src);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.capacity(), capacity);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 0);
        assert_eq!(vec[2], 9);

        for i in 0..20 {
            vec.push(i);
        }

        for i in 0..20 {
            assert_eq!(vec[i + 3], i);
        }
    }

    #[test]
    fn test_from_std_vec_dealloc() {
        let mut counter = AtomicUsize::new(0);
        let vec = HeapVec::from_std_vec(vec![IncrementOnDrop {
            counter: &mut counter,
        }]);
        drop(vec);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    std::thread_local! {
        static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);
        static RECORD_DROPS: AtomicBool = AtomicBool::new(false);
    }

    #[derive(Debug)]
    struct ZeroSized {}

    impl Drop for ZeroSized {
        fn drop(&mut self) {
            DROP_COUNTER.with(|counter| {
                RECORD_DROPS.with(|record_drops| {
                    if record_drops.load(Ordering::Relaxed) {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                });
            });
        }
    }

    impl PartialEq for ZeroSized {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    #[test]
    fn test_zero_sized_type() {
        let inline_vec = InlineVec::<ZeroSized, 4>::new();

        fn validate_vec<C>(mut vec: CraneVec<ZeroSized, C>)
        where
            C: Container<Element = ZeroSized>,
        {
            vec.push(ZeroSized {});
            vec.push(ZeroSized {});
            vec.push(ZeroSized {});
            vec.push(ZeroSized {});
            vec.push(ZeroSized {});
            assert_eq!(vec.len(), 5);
            assert_eq!(vec.capacity(), usize::MAX);
            assert_eq!(vec[0], ZeroSized {});
            assert_eq!(vec[4], ZeroSized {});
            assert_eq!(vec.get(5), None);

            for item in vec.iter() {
                assert_eq!(*item, ZeroSized {});
            }

            RECORD_DROPS.with(|record_drops| record_drops.store(true, Ordering::Relaxed));
            drop(vec);
            DROP_COUNTER.with(|counter| {
                assert_eq!(counter.load(Ordering::Relaxed), 5);
                counter.store(0, Ordering::Relaxed);
            });
            RECORD_DROPS.with(|record_drops| record_drops.store(false, Ordering::Relaxed));
        }

        validate_vec(inline_vec);

        let heap_vec = HeapVec::<ZeroSized>::new();
        validate_vec(heap_vec);

        let dyn_vec = DynVec::<ZeroSized, 4>::new();
        validate_vec(dyn_vec);
    }

    #[cfg(feature = "dropck_eyepatch")]
    #[test]
    fn test_dropck_eyepatch() {
        let mut x = 42;
        let mut vec = Vec::new();
        vec.push(&mut x);
        std::println!("{:?}", x);

        let mut x = 42;
        let mut vec = DynVec::<_, 16>::new();
        vec.push(&mut x);
        std::println!("{:?}", x);

        let mut x = 42;
        let mut vec = HeapVec::new();
        vec.push(&mut x);
        std::println!("{:?}", x);

        let mut vec = InlineVec::<_, 16>::new();
        let mut x = 42;
        vec.push(&mut x);
    }
}
