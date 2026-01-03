//! Amazon S3 integration

use crate::{CloudError, CloudUrl, DownloadOptions, ObjectMetadata, Result, UploadOptions};

/// S3 client configuration
#[derive(Debug, Clone)]
pub struct S3Config {
    /// AWS region
    pub region: String,
    /// Custom endpoint (for S3-compatible services)
    pub endpoint: Option<String>,
    /// Force path style (for MinIO, etc.)
    pub force_path_style: bool,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            region: "us-east-1".into(),
            endpoint: None,
            force_path_style: false,
        }
    }
}

/// S3 client
pub struct S3Client {
    _config: S3Config,
    #[cfg(feature = "s3")]
    client: aws_sdk_s3::Client,
}

impl S3Client {
    /// Create a new S3 client
    #[cfg(feature = "s3")]
    pub async fn new() -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = aws_sdk_s3::Client::new(&config);
        Ok(Self {
            _config: S3Config::default(),
            client,
        })
    }

    #[cfg(not(feature = "s3"))]
    pub async fn new() -> Result<Self> {
        Ok(Self {
            _config: S3Config::default(),
        })
    }

    /// Create with custom config
    #[cfg(feature = "s3")]
    pub async fn with_config(config: S3Config) -> Result<Self> {
        use aws_sdk_s3::config::Builder;

        let sdk_config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let mut builder = Builder::from(&sdk_config)
            .region(aws_sdk_s3::config::Region::new(config.region.clone()))
            .force_path_style(config.force_path_style);

        if let Some(ref endpoint) = config.endpoint {
            builder = builder.endpoint_url(endpoint);
        }

        let client = aws_sdk_s3::Client::from_conf(builder.build());
        Ok(Self { _config: config, client })
    }

    #[cfg(not(feature = "s3"))]
    pub async fn with_config(config: S3Config) -> Result<Self> {
        Ok(Self { _config: config })
    }

    /// Upload data to S3
    #[cfg(feature = "s3")]
    pub async fn upload(&self, url: &CloudUrl, data: bytes::Bytes, options: UploadOptions) -> Result<()> {
        use aws_sdk_s3::primitives::ByteStream;

        let mut builder = self.client
            .put_object()
            .bucket(&url.bucket)
            .key(&url.key)
            .body(ByteStream::from(data));

        if let Some(ref content_type) = options.content_type {
            builder = builder.content_type(content_type);
        }

        if let Some(ref storage_class) = options.storage_class {
            builder = builder.storage_class(storage_class.as_str().into());
        }

        builder.send().await.map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(())
    }

    #[cfg(not(feature = "s3"))]
    pub async fn upload(&self, _url: &CloudUrl, _data: bytes::Bytes, _options: UploadOptions) -> Result<()> {
        Err(CloudError::UnsupportedProvider("S3 feature not enabled".into()))
    }

    /// Download data from S3
    #[cfg(feature = "s3")]
    pub async fn download(&self, url: &CloudUrl, options: DownloadOptions) -> Result<bytes::Bytes> {
        let mut builder = self.client
            .get_object()
            .bucket(&url.bucket)
            .key(&url.key);

        if let (Some(start), Some(end)) = (options.range_start, options.range_end) {
            builder = builder.range(format!("bytes={}-{}", start, end));
        }

        let response = builder.send().await.map_err(|e| CloudError::Provider(e.to_string()))?;
        let data = response.body.collect().await.map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(data.into_bytes())
    }

    #[cfg(not(feature = "s3"))]
    pub async fn download(&self, _url: &CloudUrl, _options: DownloadOptions) -> Result<bytes::Bytes> {
        Err(CloudError::UnsupportedProvider("S3 feature not enabled".into()))
    }

    /// Get object metadata
    #[cfg(feature = "s3")]
    pub async fn head(&self, url: &CloudUrl) -> Result<ObjectMetadata> {
        let response = self.client
            .head_object()
            .bucket(&url.bucket)
            .key(&url.key)
            .send()
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;

        Ok(ObjectMetadata {
            key: url.key.clone(),
            size: response.content_length().unwrap_or(0) as u64,
            content_type: response.content_type().map(String::from),
            last_modified: response.last_modified().map(|t| t.secs() as u64),
            etag: response.e_tag().map(String::from),
            metadata: response.metadata().cloned().unwrap_or_default(),
        })
    }

    #[cfg(not(feature = "s3"))]
    pub async fn head(&self, _url: &CloudUrl) -> Result<ObjectMetadata> {
        Err(CloudError::UnsupportedProvider("S3 feature not enabled".into()))
    }

    /// Delete object
    #[cfg(feature = "s3")]
    pub async fn delete(&self, url: &CloudUrl) -> Result<()> {
        self.client
            .delete_object()
            .bucket(&url.bucket)
            .key(&url.key)
            .send()
            .await
            .map_err(|e| CloudError::Provider(e.to_string()))?;
        Ok(())
    }

    #[cfg(not(feature = "s3"))]
    pub async fn delete(&self, _url: &CloudUrl) -> Result<()> {
        Err(CloudError::UnsupportedProvider("S3 feature not enabled".into()))
    }

    /// List objects
    #[cfg(feature = "s3")]
    pub async fn list(&self, url: &CloudUrl, prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        let mut builder = self.client
            .list_objects_v2()
            .bucket(&url.bucket);

        if let Some(p) = prefix {
            builder = builder.prefix(p);
        }

        let response = builder.send().await.map_err(|e| CloudError::Provider(e.to_string()))?;

        let objects = response.contents()
            .iter()
            .map(|obj| ObjectMetadata {
                key: obj.key().unwrap_or("").to_string(),
                size: obj.size().unwrap_or(0) as u64,
                content_type: None,
                last_modified: obj.last_modified().map(|t| t.secs() as u64),
                etag: obj.e_tag().map(String::from),
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        Ok(objects)
    }

    #[cfg(not(feature = "s3"))]
    pub async fn list(&self, _url: &CloudUrl, _prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        Err(CloudError::UnsupportedProvider("S3 feature not enabled".into()))
    }
}
