# ADR-0003: Error Handling Strategy

## Status

Accepted

## Date

2024-02

## Context

Codec libraries encounter many error conditions:
- Bitstream parsing failures (corrupted data, unexpected values)
- Unsupported features (profiles, levels, extensions)
- Resource exhaustion (memory, file handles)
- I/O errors (file not found, permission denied)
- Configuration errors (invalid parameters)

We need an error handling strategy that:
1. Provides detailed information for debugging
2. Allows programmatic error handling
3. Is ergonomic for library users
4. Supports error context chaining
5. Performs well (errors are exceptional, not the common path)

## Decision

Use a **hierarchical error type system** with the following components:

### 1. Top-Level Error Enum

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),

    #[error("Container error: {0}")]
    Container(#[from] ContainerError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{context}")]
    WithContext {
        #[source]
        source: Box<Error>,
        context: String,
    },
}
```

### 2. Domain-Specific Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("Bitstream corruption: {message}")]
    BitstreamCorruption { message: String },

    #[error("Unsupported profile: {profile}")]
    UnsupportedProfile { profile: u8 },

    #[error("AV1 error: {0}")]
    Av1(#[from] Av1ErrorKind),
}
```

### 3. Codec-Specific Error Kinds

```rust
#[derive(Debug, thiserror::Error)]
pub enum Av1ErrorKind {
    #[error("Invalid OBU type: {0}")]
    InvalidObuType(u8),

    #[error("Sequence header missing")]
    MissingSequenceHeader,
}
```

### 4. Error Context Extension Trait

```rust
pub trait ErrorContext<T> {
    fn context(self, msg: impl Into<String>) -> Result<T>;
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}
```

### 5. Error Analysis Methods

```rust
impl Error {
    pub fn is_eof(&self) -> bool;
    pub fn is_recoverable(&self) -> bool;
    pub fn root_cause(&self) -> &Error;
}
```

## Consequences

### Positive

1. **Type-safe error handling**: Match on specific error types
   ```rust
   match result {
       Err(Error::Codec(CodecError::Av1(Av1ErrorKind::MissingSequenceHeader))) => {
           // Handle specifically
       }
       Err(e) => return Err(e),
       Ok(v) => v,
   }
   ```

2. **Error chaining**: Full context preserved
   ```rust
   decoder.decode(packet)
       .context("Failed to decode frame 42")?
   ```

3. **Performance**: `thiserror` generates efficient code, no heap allocation for simple errors

4. **Ergonomic**: `?` operator works naturally with `From` impls

5. **Debugging**: Full error chain visible in debug output

### Negative

1. **Boilerplate**: Many error types to define and maintain

2. **Learning curve**: Users must understand the hierarchy

3. **API surface**: Public error types are part of semver

### Mitigations

1. **Use `thiserror`**: Reduces boilerplate significantly

2. **Document errors**: Each error variant has clear documentation

3. **Result type alias**: `pub type Result<T> = std::result::Result<T, Error>`

## Alternatives Considered

### Alternative 1: `anyhow` for Everything

Use `anyhow::Error` as the error type.

Rejected because:
- No way to match on specific error types
- Less suitable for library APIs (better for applications)

### Alternative 2: Single Flat Error Enum

One large enum with all error variants.

Rejected because:
- Doesn't scale with 70+ crates
- Hard to add codec-specific errors without affecting all users

### Alternative 3: Trait Objects (`Box<dyn Error>`)

Use trait objects for flexibility.

Rejected because:
- No pattern matching
- Heap allocation on every error
- Less type safety

## References

- [thiserror crate](https://docs.rs/thiserror)
- [Error Handling in Rust](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [anyhow vs thiserror](https://nick.groenen.me/posts/rust-error-handling/)
