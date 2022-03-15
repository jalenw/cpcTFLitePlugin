/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.cpc.cpctfliteandroid.activity;
//package org.tensorflow.lite.examples.detection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
//import org.tensorflow.lite.examples.detection.customview.OverlayView;
//import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
//import org.tensorflow.lite.examples.detection.env.BorderedText;
//import org.tensorflow.lite.examples.detection.env.ImageUtils;
//import org.tensorflow.lite.examples.detection.env.Logger;
//import org.tensorflow.lite.examples.detection.tflite.Detector;
//import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
//import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import com.cpc.cpctfliteandroid.utils.customview.OverlayView;
import com.cpc.cpctfliteandroid.utils.customview.OverlayView.DrawCallback;
import com.cpc.cpctfliteandroid.utils.env.BorderedText;
import com.cpc.cpctfliteandroid.utils.env.ImageUtils;
import com.cpc.cpctfliteandroid.utils.env.Logger;
import com.cpc.cpctfliteandroid.tflite.Detector;
import com.cpc.cpctfliteandroid.tflite.TFLiteObjectDetectionAPIModel;
import com.cpc.cpctfliteandroid.utils.mresource.MResource;
import com.cpc.cpctfliteandroid.utils.tracking.MultiBoxTracker;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  // private static final int TF_OD_API_INPUT_SIZE = 320;
  // private static final boolean TF_OD_API_IS_QUANTIZED = false;
  // private static final boolean TF_OD_API_INCLUDE_KEYPOINTS = false;
  // private static final String TF_OD_API_MODEL_FILE = "ssdlite_mobiledet_custom_16_gpu.tflite";
  // private static final String TF_OD_API_LABELS_FILE = "labelmap_custom_16.txt";
  // private static final String TF_OD_API_FEATURE_MODEL_FILE = null;
  // private static final String TF_OD_API_FEATURE_FILE = null;

  // Configuration values for the prepackaged CenterNet model.
  private static final int TF_OD_API_INPUT_SIZE = 256;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final boolean TF_OD_API_INCLUDE_KEYPOINTS = true;
  //private static final String TF_OD_API_MODEL_FILE = "centernet_kpts_256x256.tflite";
  private static final String TF_OD_API_MODEL_FILE = "centernet_kpts_ehcm_256x256.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap_centernet.txt";
  private static final String TF_OD_API_FEATURE_MODEL_FILE = "mobilenet_v2_ep043/model.tflite";
  private static final String TF_OD_API_FEATURES_FILE = "mobilenet_v2_ep043/features.txt";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.1f;
  private static final boolean MAINTAIN_ASPECT = true;
  private static Size DESIRED_PREVIEW_SIZE = new Size(800, 600);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap rgbFrameRotationBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix frameRotationTransform;
  private Matrix cropToFrameTransform;

  protected int rotationWidth = 0;
  protected int rotationHeight = 0;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;
  
  String BrandDataString;
  JSONArray BrandDataJsonArray;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
        BrandDataString = super.getBrandDataString();
        LOGGER.d("=============================");
        LOGGER.d(BrandDataString);
        
        try {
            JSONArray tempBrandDataJsonArray = new JSONArray(BrandDataString);
            BrandDataJsonArray = new JSONArray(tempBrandDataJsonArray.get(0).toString());
            } catch (JSONException e) {
            e.printStackTrace();
        }
        
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this, BrandDataJsonArray);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              this,
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED,
              TF_OD_API_INCLUDE_KEYPOINTS,
              TF_OD_API_FEATURE_MODEL_FILE,
              TF_OD_API_FEATURES_FILE,
              MINIMUM_CONFIDENCE_TF_OD_API);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
	
	if (TF_OD_API_INCLUDE_KEYPOINTS){
      double radian = Math.toRadians(sensorOrientation);
      double sin = Math.sin(radian);
      double cos = Math.cos(radian);
      rotationWidth = (int)(previewWidth*Math.abs(cos)+previewHeight*Math.abs(sin));
      rotationHeight = (int)(previewWidth*Math.abs(sin)+previewHeight*Math.abs(cos));
      frameRotationTransform = ImageUtils.getRotationMatrix(
              previewWidth, previewHeight,
              rotationWidth, rotationHeight,
              sensorOrientation
      );
      rgbFrameRotationBitmap = Bitmap.createBitmap(rotationWidth, rotationHeight, Config.ARGB_8888);
    }
	
    //trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay = (OverlayView) findViewById(MResource.getIdByName(this, "id", "tracking_overlay"));

    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }
  
  private Bitmap getImageFromAssetsFile(String fileName){

    Bitmap image = null;
    AssetManager am = getResources().getAssets();
    try
    {
      InputStream is = am.open(fileName);
      image = BitmapFactory.decodeStream(is);
      is.close();
    }
    catch (IOException e)
    {
      e.printStackTrace();
    }

    return image;

  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    
	int new_trackingOverlay_topleft = (trackingOverlay.getBottom() - DESIRED_PREVIEW_SIZE.getWidth() * trackingOverlay.getRight() / DESIRED_PREVIEW_SIZE.getHeight()) / 2;
	if (new_trackingOverlay_topleft < 0)
		new_trackingOverlay_topleft = 0;
	trackingOverlay.setTop(new_trackingOverlay_topleft);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
	
	if (TF_OD_API_INCLUDE_KEYPOINTS){
      final Canvas canvasRotation = new Canvas(rgbFrameRotationBitmap);
      canvasRotation.drawBitmap(rgbFrameBitmap, frameRotationTransform, null);
    }
	
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap, rgbFrameRotationBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Detector.Recognition> mappedRecognitions =
                new ArrayList<Detector.Recognition>();

            for (final Detector.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    //return R.layout.tfe_od_camera_connection_fragment_tracking;
    return MResource.getIdByName(this, "layout", "tfe_od_camera_connection_fragment_tracking");
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
	if(android.os.Build.MODEL.equals("CDY-NX9B")){
		DESIRED_PREVIEW_SIZE = new Size(640, 480);
    }
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
