use qdrant_client::qdrant::PointStruct;
use serde_json::json;

use crate::embedder::FaceEmbedding;
use crate::image::FaceHash;

pub struct Face {
    pub hash: FaceHash,
    pub embedding: FaceEmbedding,
    pub celebrity: String,
    pub year_taken: Option<u16>,
}

impl From<Face> for PointStruct {
    fn from(value: Face) -> Self {
        PointStruct::new(
            value.hash.to_string(),
            value.embedding.data().to_vec(),
            json!({
                "celebrity": value.celebrity,
                "year_taken": value.year_taken
            })
            .try_into()
            .unwrap(),
        )
    }
}
