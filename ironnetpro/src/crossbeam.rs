/*
The MIT License (MIT)

Copyright (c) 2019 The Crossbeam Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.



Accessed: April 8, 2020

https://docs.rs/crossbeam/0.7.3/crossbeam/fn.scope.html
https://docs.rs/crossbeam-utils/0.7.0/src/crossbeam_utils/thread.rs.html#149-194
https://github.com/crossbeam-rs/crossbeam/blob/master/crossbeam-utils/src/sync/wait_group.rs

*/

use std::{io, mem, thread};
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex};

pub struct WaitGroup {
    inner: Arc<Inner>,
}

struct Inner {
    cvar: Condvar,
    count: Mutex<usize>,
}

impl WaitGroup {
    pub fn new() -> WaitGroup {
        WaitGroup {
            inner: Arc::new(Inner {
                cvar: Condvar::new(),
                count: Mutex::new(1),
            }),
        }
    }

	pub fn wait(self) {
        if *self.inner.count.lock().unwrap() == 1 {
            return;
        }

        let inner = self.inner.clone();
        drop(self);

        let mut count = inner.count.lock().unwrap();
        while *count > 0 {
            count = inner.cvar.wait(count).unwrap();
        }
    }
}

impl Drop for WaitGroup {
    fn drop(&mut self) {
        let mut count = self.inner.count.lock().unwrap();
        *count -= 1;

        if *count == 0 {
            self.inner.cvar.notify_all();
        }
    }
}

impl Clone for WaitGroup {
    fn clone(&self) -> WaitGroup {
        let mut count = self.inner.count.lock().unwrap();
        *count += 1;

        WaitGroup {
            inner: self.inner.clone(),
        }
    }
}

// A scope for spawning threads.
pub struct Scope<'env> {
    // The list of the thread join handles.
    handles: SharedVec<SharedOption<thread::JoinHandle<()>>>,

    // Used to wait until all subscopes all dropped.
    wait_group: WaitGroup,

    // Borrows data with invariant lifetime `'env`.
    _marker: PhantomData<&'env mut &'env ()>,
}

type SharedVec<T> = Arc<Mutex<Vec<T>>>;
type SharedOption<T> = Arc<Mutex<Option<T>>>;
use std::panic;

pub fn scope<'env, F, R>(f: F) -> thread::Result<R> where F: FnOnce(&Scope<'env>) -> R, {
    let wg = WaitGroup::new();
    let scope = Scope::<'env> {
        handles: SharedVec::default(),
        wait_group: wg.clone(),
        _marker: PhantomData,
    };

    // Execute the scoped function, but catch any panics.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&scope)));

    // Wait until all nested scopes are dropped.
    drop(scope.wait_group);
    wg.wait();

    // Join all remaining spawned threads.
    let panics: Vec<_> = {
        let mut handles = scope.handles.lock().unwrap();

        // Filter handles that haven't been joined, join them, and collect errors.
        let panics = handles
            .drain(..)
            .filter_map(|handle| handle.lock().unwrap().take())
            .filter_map(|handle| handle.join().err())
            .collect();

        panics
    };

    // If `f` has panicked, resume unwinding.
    // If any of the child threads have panicked, return the panic errors.
    // Otherwise, everything is OK and return the result of `f`.
    match result {
        Err(err) => panic::resume_unwind(err),
        Ok(res) => {
            if panics.is_empty() {
                Ok(res)
            } else {
                Err(Box::new(panics))
            }
        }
    }
}

impl<'env> Scope<'env> {
    /// Spawns a scoped thread.
    ///
    /// This method is similar to the [`spawn`] function in Rust's standard library. The difference
    /// is that this thread is scoped, meaning it's guaranteed to terminate before the scope exits,
    /// allowing it to reference variables outside the scope.
    ///
    /// The scoped thread is passed a reference to this scope as an argument, which can be used for
    /// spawning nested threads.
    ///
    /// The returned handle can be used to manually join the thread before the scope exits.
    ///
    /// [`spawn`]: https://doc.rust-lang.org/std/thread/fn.spawn.html
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::thread;
    ///
    /// thread::scope(|s| {
    ///     let handle = s.spawn(|_| {
    ///         println!("A child thread is running");
    ///         42
    ///     });
    ///
    ///     // Join the thread and retrieve its result.
    ///     let res = handle.join().unwrap();
    ///     assert_eq!(res, 42);
    /// }).unwrap();
    /// ```
    pub fn spawn<'scope, F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope>
    where
        F: FnOnce(&Scope<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        self.builder().spawn(f).unwrap()
    }

    /// Creates a builder that can configure a thread before spawning.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::thread;
    /// use std::thread::current;
    ///
    /// thread::scope(|s| {
    ///     s.builder()
    ///         .spawn(|_| println!("A child thread is running"))
    ///         .unwrap();
    /// }).unwrap();
    /// ```
    pub fn builder<'scope>(&'scope self) -> ScopedThreadBuilder<'scope, 'env> {
        ScopedThreadBuilder {
            scope: self,
            builder: thread::Builder::new(),
        }
    }
}
pub struct ScopedThreadBuilder<'scope, 'env: 'scope> {
    scope: &'scope Scope<'env>,
    builder: thread::Builder,
}

impl<'scope, 'env> ScopedThreadBuilder<'scope, 'env> {
    /// Sets the name for the new thread.
    ///
    /// The name must not contain null bytes. For more information about named threads, see
    /// [here][naming-threads].
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::thread;
    /// use std::thread::current;
    ///
    /// thread::scope(|s| {
    ///     s.builder()
    ///         .name("my thread".to_string())
    ///         .spawn(|_| assert_eq!(current().name(), Some("my thread")))
    ///         .unwrap();
    /// }).unwrap();
    /// ```
    ///
    /// [naming-threads]: https://doc.rust-lang.org/std/thread/index.html#naming-threads
    pub fn name(mut self, name: String) -> ScopedThreadBuilder<'scope, 'env> {
        self.builder = self.builder.name(name);
        self
    }

    /// Sets the size of the stack for the new thread.
    ///
    /// The stack size is measured in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::thread;
    ///
    /// thread::scope(|s| {
    ///     s.builder()
    ///         .stack_size(32 * 1024)
    ///         .spawn(|_| println!("Running a child thread"))
    ///         .unwrap();
    /// }).unwrap();
    /// ```
    pub fn stack_size(mut self, size: usize) -> ScopedThreadBuilder<'scope, 'env> {
        self.builder = self.builder.stack_size(size);
        self
    }

    /// Spawns a scoped thread with this configuration.
    ///
    /// The scoped thread is passed a reference to this scope as an argument, which can be used for
    /// spawning nested threads.
    ///
    /// The returned handle can be used to manually join the thread before the scope exits.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::thread;
    ///
    /// thread::scope(|s| {
    ///     let handle = s.builder()
    ///         .spawn(|_| {
    ///             println!("A child thread is running");
    ///             42
    ///         })
    ///         .unwrap();
    ///
    ///     // Join the thread and retrieve its result.
    ///     let res = handle.join().unwrap();
    ///     assert_eq!(res, 42);
    /// }).unwrap();
    /// ```
    pub fn spawn<F, T>(self, f: F) -> io::Result<ScopedJoinHandle<'scope>>
    where
        F: FnOnce(&Scope<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        // The result of `f` will be stored here.
        let result = SharedOption::default();

        // Spawn the thread and grab its join handle and thread handle.
        let handle = {
            let result = Arc::clone(&result);

            // A clone of the scope that will be moved into the new thread.
            let scope = Scope::<'env> {
                handles: Arc::clone(&self.scope.handles),
                wait_group: self.scope.wait_group.clone(),
                _marker: PhantomData,
            };

            // Spawn the thread.
            let handle = {
                let closure = move || {
                    // Make sure the scope is inside the closure with the proper `'env` lifetime.
                    let scope: Scope<'env> = scope;

                    // Run the closure.
                    let res = f(&scope);

                    // Store the result if the closure didn't panic.
                    *result.lock().unwrap() = Some(res);
                };

                // Change the type of `closure` from `FnOnce() -> T` to `FnMut() -> T`.
                let mut closure = Some(closure);
                let closure = move || closure.take().unwrap()();

                // Allocate `clsoure` on the heap and erase the `'env` bound.
                let closure: Box<dyn FnMut() + Send + 'env> = Box::new(closure);
                let closure: Box<dyn FnMut() + Send + 'static> = unsafe { mem::transmute(closure) };

                // Finally, spawn the closure.
                let mut closure = closure;
                self.builder.spawn(move || closure())?
            };

            //let thread = handle.thread().clone();
            let handle = Arc::new(Mutex::new(Some(handle)));
            handle
        };

        // Add the handle to the shared list of join handles.
        self.scope.handles.lock().unwrap().push(Arc::clone(&handle));

        Ok(ScopedJoinHandle {
            //handle,
            //result,
            //thread,
            _marker: PhantomData,
        })
    }
}

unsafe impl<'scope> Send for ScopedJoinHandle<'scope> {}
unsafe impl<'scope> Sync for ScopedJoinHandle<'scope> {}

/// A handle that can be used to join its scoped thread.
pub struct ScopedJoinHandle<'scope> {//<'scope, T> {
    /// A join handle to the spawned thread.
   // handle: SharedOption<thread::JoinHandle<()>>,

    /// Holds the result of the inner closure.
    //result: SharedOption<T>,

    /// A handle to the the spawned thread.
    //thread: thread::Thread,

    /// Borrows the parent scope with lifetime `'scope`.
    _marker: PhantomData<&'scope ()>,
}

