use crate::engine::{session_from_bytes, session_from_path};
use crate::models::SuperResolution;
use ndarray::{Array, Ix4};
use opencv::{core, core::Mat, dnn, imgproc, prelude::*};
use ort::{Result as OrtResult, session::Session, value::Tensor};
use std::path::Path;

pub struct RealEsrgan {
    session: Session,
}

impl RealEsrgan {
    pub fn from_path(path: &Path) -> OrtResult<Self> {
        let session = session_from_path(path)?;
        Ok(Self { session })
    }

    pub fn from_bytes(bytes: &[u8]) -> OrtResult<Self> {
        let session = session_from_bytes(bytes)?;
        Ok(Self { session })
    }

    fn scale_from_depth(depth: i32) -> f64 {
        match depth {
            core::CV_8U => 1.0 / 255.0,
            core::CV_16U | core::CV_16S => 1.0 / 65535.0,
            _ => 1.0 / 255.0,
        }
    }

    fn ensure_bgr(input: &Mat, channels: i32) -> Mat {
        match channels {
            4 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_BGRA2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).unwrap();
                bgr
            }
            1 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).unwrap();
                bgr
            }
            _ => input.clone(),
        }
    }

    /// Ensure input size is valid for model reshape by padding to even width/height.
    fn pad_even(input: &Mat) -> (Mat, i32, i32) {
        let w = input.cols();
        let h = input.rows();
        let pad_right = if w % 2 != 0 { 1 } else { 0 };
        let pad_bottom = if h % 2 != 0 { 1 } else { 0 };
        if pad_right == 0 && pad_bottom == 0 {
            return (input.clone(), 0, 0);
        }
        let mut padded = Mat::default();
        // Use core::copy_make_border with REFLECT_101 to avoid visible seams
        core::copy_make_border(
            input,
            &mut padded,
            0,
            pad_bottom,
            0,
            pad_right,
            core::BORDER_REFLECT_101,
            core::Scalar::default(),
        )
        .unwrap();
        (padded, pad_right, pad_bottom)
    }

    fn compose_bgra(out_bgr: &Mat, input: &Mat, depth: i32, w_out: i32, h_out: i32) -> Mat {
        let mut alpha = Mat::default();
        core::extract_channel(input, &mut alpha, 3).unwrap();

        let alpha_u8 = if depth == core::CV_8U {
            alpha
        } else {
            let mut tmp = Mat::default();
            alpha.convert_to(&mut tmp, core::CV_8U, 255.0 / 65535.0, 0.0).unwrap();
            tmp
        };

        let mut alpha_up = Mat::default();
        imgproc::resize(&alpha_u8, &mut alpha_up, core::Size::new(w_out, h_out), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

        let mut out_rgba = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC4, core::Scalar::default()).unwrap();

        let mut src = core::Vector::<Mat>::new();
        src.push(out_bgr.clone());
        src.push(alpha_up);
        let from_to = [0, 0, 1, 1, 2, 2, 3, 3];
        core::mix_channels(&src, &mut out_rgba, &from_to).unwrap();
        out_rgba
    }
}

impl SuperResolution for RealEsrgan {
    fn run(&mut self, input: Mat) -> OrtResult<Mat> {
        let (h, w, channels, depth) = (input.rows(), input.cols(), input.channels(), input.depth());
        let scale = Self::scale_from_depth(depth);
        let bgr_input = Self::ensure_bgr(&input, channels);
        // Pad to even dimensions to satisfy ONNX reshape nodes that assume even H/W
        let (padded, _pad_right, _pad_bottom) = Self::pad_even(&bgr_input);
        let (w_pad, h_pad) = (padded.cols(), padded.rows());

        let blob = dnn::blob_from_image(
            &padded,
            scale,
            core::Size::new(w_pad, h_pad),
            core::Scalar::default(),
            true,
            false,
            core::CV_32F,
        )
        .unwrap();

        let blob_slice: &[f32] = blob.data_typed().unwrap();
        let input_array = Array::from_shape_vec((1, 3, h_pad as usize, w_pad as usize), blob_slice.to_vec()).unwrap();
        let input_tensor = Tensor::from_array(input_array)?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let out_view = outputs[0].try_extract_array::<f32>().unwrap();
        let out4 = out_view.into_dimensionality::<Ix4>().unwrap();
        let (h_out, w_out) = (out4.shape()[2] as i32, out4.shape()[3] as i32);

        let mut out_mat = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC3, core::Scalar::default()).unwrap();

        let buf = out_mat.data_bytes_mut().unwrap();
        let w_out_usize = w_out as usize;
        // Avoid thread oversubscription: write in a single-threaded loop
        for (y, row) in buf.chunks_mut(w_out_usize * 3).enumerate() {
            for x in 0..w_out_usize {
                let r = out4[[0, 0, y, x]].clamp(0.0, 1.0);
                let g = out4[[0, 1, y, x]].clamp(0.0, 1.0);
                let b = out4[[0, 2, y, x]].clamp(0.0, 1.0);

                let idx = x * 3;
                row[idx] = (b * 255.0).round() as u8;
                row[idx + 1] = (g * 255.0).round() as u8;
                row[idx + 2] = (r * 255.0).round() as u8;
            }
        }

        // If we padded the input, crop the output back to the original scaled size.
        // Estimate scale factor from output vs padded input size.
        let scale_est_w = (w_out as f64) / (w_pad as f64);
        let scale_est_h = (h_out as f64) / (h_pad as f64);
        let scale_est = ((scale_est_w + scale_est_h) / 2.0).round() as i32; // assume isotropic scaling (2 or 4)
        let target_w = w * scale_est;
        let target_h = h * scale_est;

        let out_cropped = if target_w < w_out || target_h < h_out {
            let roi = core::Rect::new(0, 0, target_w.max(1), target_h.max(1));
            let m = out_mat.roi(roi).unwrap();
            // Convert ROI view to an owned Mat
            m.try_clone().unwrap()
        } else {
            out_mat
        };

        if channels == 4 {
            // Compose alpha and then crop if necessary to match target size
            let out_rgba_full = Self::compose_bgra(&out_cropped, &input, depth, out_cropped.cols(), out_cropped.rows());
            Ok(out_rgba_full)
        } else {
            Ok(out_cropped)
        }
    }
}
