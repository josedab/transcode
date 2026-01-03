# transcode-cloud

Cloud storage integration for the transcode project, providing unified access to Amazon S3, Google Cloud Storage, and Azure Blob Storage.

## Features

- **Multi-provider support**: S3, GCS, and Azure Blob Storage via unified API
- **URL parsing**: Parse and construct cloud URLs (`s3://`, `gs://`, `az://`)
- **File operations**: Upload, download, delete, list, and check existence
- **Streaming support**: Range requests for partial downloads
- **Multipart uploads**: Configurable thresholds and part sizes
- **S3-compatible services**: Support for MinIO and other S3-compatible endpoints

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-cloud = { path = "../transcode-cloud", features = ["s3", "gcs"] }
```

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `s3` | Amazon S3 support | `aws-sdk-s3`, `aws-config` |
| `gcs` | Google Cloud Storage support | `cloud-storage` |

## Key Types

| Type | Description |
|------|-------------|
| `CloudClient` | Unified client for all providers |
| `CloudUrl` | Parsed cloud storage URL |
| `CloudProvider` | Enum: `S3`, `Gcs`, `Azure` |
| `S3Client` | Direct S3 client with custom config |
| `GcsClient` | Direct GCS client with custom config |
| `UploadOptions` | Content type, storage class, metadata |
| `DownloadOptions` | Range start/end for partial downloads |
| `ObjectMetadata` | Key, size, content type, ETag, etc. |
| `CloudError` | Error type for cloud operations |

## Usage

### Parsing Cloud URLs

```rust
use transcode_cloud::CloudUrl;

let url = CloudUrl::parse("s3://my-bucket/path/to/video.mp4")?;
assert_eq!(url.bucket, "my-bucket");
assert_eq!(url.key, "path/to/video.mp4");
```

### Using the Unified Client

```rust
use transcode_cloud::{CloudClient, CloudUrl, UploadOptions};
use std::path::Path;

#[tokio::main]
async fn main() -> transcode_cloud::Result<()> {
    let client = CloudClient::new().await?;
    let url = CloudUrl::parse("s3://my-bucket/output.mp4")?;

    // Upload a file
    client.upload_file(&url, Path::new("/local/video.mp4")).await?;

    // Download a file
    client.download_file(&url, Path::new("/local/downloaded.mp4")).await?;

    Ok(())
}
```

### Using S3 with Custom Configuration

```rust
use transcode_cloud::{S3Client, S3Config, CloudUrl, UploadOptions};

let config = S3Config {
    region: "us-west-2".into(),
    endpoint: Some("http://localhost:9000".into()), // MinIO
    force_path_style: true,
};

let client = S3Client::with_config(config).await?;
let url = CloudUrl::parse("s3://bucket/key.mp4")?;
let data = bytes::Bytes::from(vec![0u8; 1024]);

client.upload(&url, data, UploadOptions::default()).await?;
```

### Listing Objects

```rust
use transcode_cloud::{CloudClient, CloudUrl};

let client = CloudClient::new().await?;
let url = CloudUrl::parse("s3://my-bucket/")?;

let objects = client.list(&url, Some("videos/")).await?;
for obj in objects {
    println!("{}: {} bytes", obj.key, obj.size);
}
```

## Error Handling

The crate uses `CloudError` for all operations:

- `InvalidUrl` - Malformed cloud URL
- `NotFound` - Object does not exist
- `AccessDenied` - Permission denied
- `UnsupportedProvider` - Provider feature not enabled
- `Network` - Network connectivity issues
- `Provider` - Provider-specific errors

## License

See the workspace root for license information.
