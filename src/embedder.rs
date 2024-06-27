use anyhow::Result;
use opencv::{
    core::{Mat, Scalar},
    dnn::{self, Net, NetTrait},
    prelude::MatTraitConst,
};

use crate::image::{FaceLocation, FaceRegion, Image};

pub struct FaceEmbedder(Net);

impl FaceEmbedder {
    pub fn new() -> Result<Self> {
        let embedder_model = dnn::read_net_from_torch_def("./models/openface.nn4.small2.v1.t7")?;

        Ok(Self(embedder_model))
    }

    pub fn embed(&mut self, face: &FaceRegion) -> Result<FaceEmbedding> {
        let img_blob = face.to_blob()?;
        self.0.set_input_def(&img_blob)?;

        let mut output_blobs = Mat::default();

        self.0.forward_layer_def(&mut output_blobs)?;

        let mut data = [0f32; 128];

        for i in 0..128 {
            let value = output_blobs.at_2d::<f32>(0, i)?;
            data[i as usize] = *value;
        }

        Ok(FaceEmbedding(data))
    }
}

#[derive(Debug)]
pub struct FaceEmbedding([f32; 128]);

impl FaceEmbedding {
    pub fn data(&self) -> [f32; 128] {
        self.0
    }
}

pub struct FaceDetector(Net);

impl FaceDetector {
    pub fn new() -> Result<Self> {
        const PROTOTXT: &str = "./models/deploy.prototxt";
        const CAFFE_MODEL: &str = "./models/res10_300x300_ssd_iter_140000.caffemodel";

        let detector_model = dnn::read_net_from_caffe(PROTOTXT, CAFFE_MODEL)?;

        Ok(Self(detector_model))
    }

    pub fn detect_best(&mut self, image: &Image) -> Result<Option<FaceLocation>> {
        let results = self.detect(image)?;

        match results.get(0) {
            Some((confidence, location)) if *confidence > 0.9 => Ok(Some(*location)),
            _ => Ok(None),
        }
    }

    pub fn detect(&mut self, image: &Image) -> Result<Vec<(f32, FaceLocation)>> {
        let img_blob = image.to_blob()?;

        self.0
            .set_input(&img_blob, "data", 1.0, Scalar::default())?;

        let mut output_blobs = Mat::default();

        self.0.forward_layer_def(&mut output_blobs)?;

        let mut results = Vec::new();

        for i in 0..200 {
            let confidence = output_blobs.at_nd::<f32>(&[0, 0, i, 2])?;
            let x1 = output_blobs.at_nd::<f32>(&[0, 0, i, 3])?;
            let y1 = output_blobs.at_nd::<f32>(&[0, 0, i, 4])?;
            let x2 = output_blobs.at_nd::<f32>(&[0, 0, i, 5])?;
            let y2 = output_blobs.at_nd::<f32>(&[0, 0, i, 6])?;

            let face_location = FaceLocation::new(*x1, *y1, x2 - x1, y2 - y1);

            results.push((*confidence, face_location));
        }

        Ok(results)
    }
}
