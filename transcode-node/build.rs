//! Build script for napi-rs.
//!
//! This script is required by napi-rs to generate the necessary
//! Node.js binding code during the build process.

extern crate napi_build;

fn main() {
    napi_build::setup();
}
