mod embedder;
mod face;
mod image;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, Distance, PointStruct, VectorParams, VectorsConfig};
use serde_json::json;

use self::embedder::{FaceDetector, FaceEmbedder};
use self::face::Face;
use self::image::Image;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Add {
        #[arg(short, long)]
        path: PathBuf,
        #[arg(short, long)]
        celebrity: String,
    },
    Guess {
        #[arg(short, long)]
        path: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Add { path, celebrity } => {
            println!("add: {:?} {:?}", path, celebrity);

            let mut face_detector = FaceDetector::new()?;
            let mut face_embedder = FaceEmbedder::new()?;

            let image = Image::load(&path.to_str().unwrap())?;
            let location = face_detector.detect_best(&image)?;

            if let Some(location) = location {
                let region = image.extract_face(location)?;

                let face_hash = region.hash()?;
                let embedding = face_embedder.embed(&region)?;

                let face = Face {
                    hash: face_hash,
                    embedding,
                    celebrity,
                    year_taken: None,
                };

                let client = QdrantClient::from_url("http://localhost:6334").build()?;

                let has_collection = client.collection_exists("faces").await?;

                if !has_collection {
                    client
                        .create_collection(&CreateCollection {
                            collection_name: "faces".into(),
                            vectors_config: Some(VectorsConfig {
                                config: Some(Config::Params(VectorParams {
                                    size: 128,
                                    distance: Distance::Cosine.into(),
                                    ..Default::default()
                                })),
                            }),
                            ..Default::default()
                        })
                        .await?;
                }

                let point = PointStruct::from(face);

                let operation_info = client
                    .upsert_points_blocking("faces".to_string(), None, vec![point], None)
                    .await?;

                dbg!(operation_info);
            }
        }

        Commands::Guess { path } => {
            println!("guess: {:?}", path);
        }
    }

    Ok(())
}
