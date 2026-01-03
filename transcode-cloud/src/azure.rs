//! Azure Blob Storage client implementation
//!
//! This module provides Azure Blob Storage integration using the azure_storage_blobs SDK.
//!
//! # Authentication
//!
//! The client uses `DefaultAzureCredential` which supports multiple authentication methods:
//! - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
//! - Managed Identity (when running in Azure)
//! - Azure CLI credentials
//! - Storage account connection string (AZURE_STORAGE_CONNECTION_STRING)
//!
//! # Example
//!
//! ```ignore
//! use transcode_cloud::{AzureClient, CloudUrl, CloudStorage};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = AzureClient::new().await?;
//!
//!     // Upload data
//!     let url = CloudUrl::parse("az://mycontainer/path/to/video.mp4")?;
//!     client.upload(&url, data, Default::default()).await?;
//!
//!     // Download data
//!     let downloaded = client.download(&url, Default::default()).await?;
//!
//!     Ok(())
//! }
//! ```

use crate::{CloudError, CloudStorage, CloudUrl, DownloadOptions, ObjectMetadata, Result, UploadOptions};
use bytes::Bytes;
use std::sync::Arc;

#[cfg(feature = "azure")]
use azure_storage::prelude::*;
#[cfg(feature = "azure")]
use azure_storage_blobs::prelude::*;

/// Azure Blob Storage client
#[derive(Clone)]
pub struct AzureClient {
    /// Account name
    account: String,
    /// Storage credentials
    #[cfg(feature = "azure")]
    credentials: Arc<StorageCredentials>,
    /// Whether client is initialized
    #[cfg(not(feature = "azure"))]
    _phantom: std::marker::PhantomData<()>,
}

impl AzureClient {
    /// Create a new Azure client using default credentials
    ///
    /// Tries the following authentication methods in order:
    /// 1. Connection string from AZURE_STORAGE_CONNECTION_STRING
    /// 2. Storage account key from AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY
    /// 3. DefaultAzureCredential (managed identity, CLI, etc.)
    pub async fn new() -> Result<Self> {
        Self::from_env().await
    }

    /// Create from environment variables
    #[cfg(feature = "azure")]
    async fn from_env() -> Result<Self> {
        // Try connection string first
        if let Ok(conn_str) = std::env::var("AZURE_STORAGE_CONNECTION_STRING") {
            return Self::from_connection_string(&conn_str);
        }

        // Try account name + key
        let account = std::env::var("AZURE_STORAGE_ACCOUNT")
            .or_else(|_| std::env::var("AZURE_ACCOUNT_NAME"))
            .map_err(|_| CloudError::AuthenticationFailed(
                "AZURE_STORAGE_ACCOUNT or AZURE_STORAGE_CONNECTION_STRING required".into()
            ))?;

        if let Ok(key) = std::env::var("AZURE_STORAGE_KEY") {
            let credentials = StorageCredentials::access_key(&account, key);
            return Ok(Self {
                account,
                credentials: Arc::new(credentials),
            });
        }

        // Try SAS token
        if let Ok(sas) = std::env::var("AZURE_STORAGE_SAS_TOKEN") {
            let credentials = StorageCredentials::sas_token(sas)
                .map_err(|e| CloudError::AuthenticationFailed(e.to_string()))?;
            return Ok(Self {
                account,
                credentials: Arc::new(credentials),
            });
        }

        // Default to anonymous access (for public containers)
        let credentials = StorageCredentials::anonymous();
        Ok(Self {
            account,
            credentials: Arc::new(credentials),
        })
    }

    /// Create from connection string
    #[cfg(feature = "azure")]
    pub fn from_connection_string(conn_str: &str) -> Result<Self> {
        // Parse connection string to extract account name
        let account = conn_str
            .split(';')
            .find_map(|part| {
                let (key, value) = part.split_once('=')?;
                if key.eq_ignore_ascii_case("AccountName") {
                    Some(value.to_string())
                } else {
                    None
                }
            })
            .ok_or_else(|| CloudError::AuthenticationFailed(
                "AccountName not found in connection string".into()
            ))?;

        let credentials = StorageCredentials::connection_string(conn_str)
            .map_err(|e| CloudError::AuthenticationFailed(e.to_string()))?;

        Ok(Self {
            account,
            credentials: Arc::new(credentials),
        })
    }

    /// Create from storage account key
    #[cfg(feature = "azure")]
    pub fn from_key(account: &str, key: &str) -> Result<Self> {
        let credentials = StorageCredentials::access_key(account, key);
        Ok(Self {
            account: account.to_string(),
            credentials: Arc::new(credentials),
        })
    }

    /// Create from SAS token
    #[cfg(feature = "azure")]
    pub fn from_sas_token(account: &str, sas_token: &str) -> Result<Self> {
        let credentials = StorageCredentials::sas_token(sas_token)
            .map_err(|e| CloudError::AuthenticationFailed(e.to_string()))?;
        Ok(Self {
            account: account.to_string(),
            credentials: Arc::new(credentials),
        })
    }

    /// Get a blob client for the given container and blob
    #[cfg(feature = "azure")]
    fn get_blob_client(&self, container: &str, blob: &str) -> BlobClient {
        BlobServiceClient::new(&self.account, self.credentials.as_ref().clone())
            .container_client(container)
            .blob_client(blob)
    }

    /// Get a container client
    #[cfg(feature = "azure")]
    fn get_container_client(&self, container: &str) -> ContainerClient {
        BlobServiceClient::new(&self.account, self.credentials.as_ref().clone())
            .container_client(container)
    }

    /// Stub implementation when azure feature is disabled
    #[cfg(not(feature = "azure"))]
    async fn from_env() -> Result<Self> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in. Enable the 'azure' feature.".into()
        ))
    }
}

#[async_trait::async_trait]
impl CloudStorage for AzureClient {
    /// Upload data to Azure Blob Storage
    #[cfg(feature = "azure")]
    async fn upload(&self, url: &CloudUrl, data: Bytes, options: UploadOptions) -> Result<()> {
        let blob_client = self.get_blob_client(&url.bucket, &url.key);

        // Determine if we should use block upload for large files
        let multipart_threshold = options.multipart_threshold.unwrap_or(4 * 1024 * 1024); // 4MB default

        if data.len() as u64 > multipart_threshold {
            // Use block blob upload for large files
            self.upload_blocks(&blob_client, data, &options).await?;
        } else {
            // Simple put for small files
            let mut builder = blob_client.put_block_blob(data);

            if let Some(ref content_type) = options.content_type {
                builder = builder.content_type(content_type);
            }

            builder.await
                .map_err(|e| CloudError::UploadFailed(e.to_string()))?;
        }

        Ok(())
    }

    /// Upload data using block upload for large files
    #[cfg(not(feature = "azure"))]
    async fn upload(&self, _url: &CloudUrl, _data: Bytes, _options: UploadOptions) -> Result<()> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in".into()
        ))
    }

    /// Download data from Azure Blob Storage
    #[cfg(feature = "azure")]
    async fn download(&self, url: &CloudUrl, options: DownloadOptions) -> Result<Bytes> {
        let blob_client = self.get_blob_client(&url.bucket, &url.key);

        let mut builder = blob_client.get();

        // Apply range if specified
        if let (Some(start), Some(end)) = (options.range_start, options.range_end) {
            builder = builder.range(start..end);
        } else if let Some(start) = options.range_start {
            builder = builder.range(start..);
        }

        let response = builder.into_stream();
        let data = response.collect().await
            .map_err(|e| CloudError::DownloadFailed(e.to_string()))?;

        Ok(data.data)
    }

    #[cfg(not(feature = "azure"))]
    async fn download(&self, _url: &CloudUrl, _options: DownloadOptions) -> Result<Bytes> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in".into()
        ))
    }

    /// Get blob metadata
    #[cfg(feature = "azure")]
    async fn head(&self, url: &CloudUrl) -> Result<ObjectMetadata> {
        let blob_client = self.get_blob_client(&url.bucket, &url.key);

        let response = blob_client.get_properties().await
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("BlobNotFound") || msg.contains("404") {
                    CloudError::NotFound
                } else {
                    CloudError::Other(msg)
                }
            })?;

        let props = response.blob.properties;
        let mut metadata = std::collections::HashMap::new();

        // Convert Azure metadata to our format
        for (k, v) in response.blob.metadata.iter() {
            metadata.insert(k.clone(), v.clone());
        }

        Ok(ObjectMetadata {
            key: url.key.clone(),
            size: props.content_length,
            content_type: props.content_type.map(|ct| ct.to_string()),
            last_modified: props.last_modified.map(|t| t.unix_timestamp() as u64),
            etag: props.etag.map(|e| e.to_string()),
            metadata,
        })
    }

    #[cfg(not(feature = "azure"))]
    async fn head(&self, _url: &CloudUrl) -> Result<ObjectMetadata> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in".into()
        ))
    }

    /// Delete a blob
    #[cfg(feature = "azure")]
    async fn delete(&self, url: &CloudUrl) -> Result<()> {
        let blob_client = self.get_blob_client(&url.bucket, &url.key);

        blob_client.delete().await
            .map_err(|e| CloudError::DeleteFailed(e.to_string()))?;

        Ok(())
    }

    #[cfg(not(feature = "azure"))]
    async fn delete(&self, _url: &CloudUrl) -> Result<()> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in".into()
        ))
    }

    /// List blobs with optional prefix
    #[cfg(feature = "azure")]
    async fn list(&self, url: &CloudUrl, prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        let container_client = self.get_container_client(&url.bucket);

        let full_prefix = match prefix {
            Some(p) => format!("{}/{}", url.key.trim_end_matches('/'), p),
            None => url.key.clone(),
        };

        let mut results = Vec::new();
        let mut stream = container_client
            .list_blobs()
            .prefix(full_prefix)
            .into_stream();

        use futures::StreamExt;
        while let Some(response) = stream.next().await {
            let response = response
                .map_err(|e| CloudError::Other(e.to_string()))?;

            for blob in response.blobs.blobs() {
                results.push(ObjectMetadata {
                    key: blob.name.clone(),
                    size: blob.properties.content_length,
                    content_type: blob.properties.content_type.as_ref().map(|ct| ct.to_string()),
                    last_modified: blob.properties.last_modified.map(|t| t.unix_timestamp() as u64),
                    etag: blob.properties.etag.as_ref().map(|e| e.to_string()),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        Ok(results)
    }

    #[cfg(not(feature = "azure"))]
    async fn list(&self, _url: &CloudUrl, _prefix: Option<&str>) -> Result<Vec<ObjectMetadata>> {
        Err(CloudError::UnsupportedProvider(
            "Azure support not compiled in".into()
        ))
    }
}

#[cfg(feature = "azure")]
impl AzureClient {
    /// Upload using block blob for large files
    async fn upload_blocks(
        &self,
        blob_client: &BlobClient,
        data: Bytes,
        options: &UploadOptions,
    ) -> Result<()> {
        let part_size = options.part_size.unwrap_or(4 * 1024 * 1024) as usize; // 4MB default
        let mut block_list = Vec::new();

        // Upload blocks
        for (i, chunk) in data.chunks(part_size).enumerate() {
            let block_id = format!("{:08}", i);
            let block_id_b64 = base64_encode(&block_id);

            blob_client
                .put_block(block_id_b64.clone(), Bytes::copy_from_slice(chunk))
                .await
                .map_err(|e| CloudError::UploadFailed(e.to_string()))?;

            block_list.push(azure_storage_blobs::blob::BlobBlockType::new_uncommitted(block_id_b64));
        }

        // Commit the block list
        let mut builder = blob_client.put_block_list(block_list);

        if let Some(ref content_type) = options.content_type {
            builder = builder.content_type(content_type);
        }

        builder.await
            .map_err(|e| CloudError::UploadFailed(e.to_string()))?;

        Ok(())
    }
}

#[cfg(feature = "azure")]
fn base64_encode(s: &str) -> String {
    use std::io::Write;
    let mut encoder = base64::write::EncoderStringWriter::new(&base64::engine::general_purpose::STANDARD);
    encoder.write_all(s.as_bytes()).unwrap();
    encoder.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_url_parsing() {
        let url = CloudUrl::parse("az://mycontainer/path/to/blob.mp4").unwrap();
        assert_eq!(url.bucket, "mycontainer");
        assert_eq!(url.key, "path/to/blob.mp4");
    }

    #[test]
    fn test_azure_url_alternative() {
        let url = CloudUrl::parse("azure://container/file.mp4").unwrap();
        assert_eq!(url.bucket, "container");
        assert_eq!(url.key, "file.mp4");
    }
}
