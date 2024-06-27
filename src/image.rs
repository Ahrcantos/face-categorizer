use anyhow::Result;
use base64::prelude::*;
use opencv::{
    core::{Mat, MatTraitConst, Rect, Scalar, Size},
    dnn, img_hash, imgcodecs, imgproc,
};

pub struct Image(Mat);

impl Image {
    pub fn load(path: &str) -> Result<Self> {
        let size = Size {
            width: 300,
            height: 300,
        };
        let img = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;

        let mut img_resized = Mat::default();
        imgproc::resize(&img, &mut img_resized, size, 0.0, 0.0, imgproc::INTER_AREA)?;

        Ok(Self(img_resized))
    }

    pub fn extract_face(&self, location: FaceLocation) -> Result<FaceRegion> {
        let (x, y, width, height) = location.coords();
        let rect = Rect::new(x, y, width, height);
        let roi = Mat::roi(&self.0, rect)?;

        let size = Size {
            width: 96,
            height: 96,
        };

        let mut roi_resized = Mat::default();
        imgproc::resize(&roi, &mut roi_resized, size, 0.0, 0.0, imgproc::INTER_CUBIC)?;

        Ok(FaceRegion(roi_resized))
    }

    pub fn to_blob(&self) -> Result<Mat> {
        let size = Size {
            width: 300,
            height: 300,
        };
        let mean = Scalar::new(0.0, 0.0, 0.0, 0.0);
        let blob = dnn::blob_from_image(&self.0, 1.0, size, mean, false, false, 0)?;

        Ok(blob)
    }
}

pub struct FaceHash([u8; 8]);

impl ToString for FaceHash {
    fn to_string(&self) -> String {
        use uuid::Uuid;

        let id = Uuid::new_v3(&Uuid::NAMESPACE_OID, &self.0);

        id.to_string()
    }
}

pub struct FaceRegion(Mat);

impl FaceRegion {
    pub fn to_blob(&self) -> Result<Mat> {
        let size = Size {
            width: 96,
            height: 96,
        };
        let mean = Scalar::new(0.0, 0.0, 0.0, 0.0);
        let blob = dnn::blob_from_image(&self.0, 1.0, size, mean, false, false, 0)?;

        Ok(blob)
    }

    pub fn hash(&self) -> Result<FaceHash> {
        let mut output = Mat::default();
        img_hash::p_hash(&self.0, &mut output)?;

        let mut hash = [0u8; 8];

        for i in 0..8i32 {
            hash[i as usize] = *output.at_2d(0, i)?;
        }

        Ok(FaceHash(hash))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FaceLocation {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl FaceLocation {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl FaceLocation {
    pub fn coords(&self) -> (i32, i32, i32, i32) {
        let x: i32 = (300.0 * self.x) as i32;
        let y: i32 = (300.0 * self.y) as i32;
        let width: i32 = (300.0 * self.width) as i32;
        let height: i32 = (300.0 * self.height) as i32;

        (x, y, width, height)
    }
}
