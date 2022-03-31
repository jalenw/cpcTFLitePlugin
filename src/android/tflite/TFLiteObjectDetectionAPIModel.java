/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.cpc.cpctfliteandroid.tflite;
//package org.tensorflow.lite.examples.detection.tflite;

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Environment;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithInterpreter";

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 1;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private boolean includeKeypoints;
  // outputKeypoints: array of shape [Batchsize, NUM_DETECTIONS, 4, 2]
  // contains the location of detected keypoints
  private float[][][][] outputKeypoints;
  // outputKeypointScores: array of shape [Batchsize, NUM_DETECTIONS, 4]
  // contains the scores of detected keypoints
  private float[][][] outputKeypointScores;
  private float[][] outputFeatureScores;
  private float[][] outputFeatures;

  //private GpuDelegate gpuDelegate = null;

  private ByteBuffer imgData;

  private MappedByteBuffer tfLiteModel;
  private Interpreter.Options tfLiteOptions;
  private Interpreter tfLite;

  private static final int FEATURE_MODEL_INPUT_SIZE = 224;
  private static final int FEATURE_MODEL_CLASS_NUM = 42;
  private static final int FEATURE_MODEL_FEATURE_SIZE = 128;
  private static final float FEATURE_MIN_DISTANCE = 1.2f;
  private float detectMinConfidence = 1.0f;
  private byte[] featureMatIntValues;
  private final List<String> featureLabels = new ArrayList<>();
  private final List<float[]> featureValues = new ArrayList<>();
  //private GpuDelegate featureGpuDelegate = null;
  private ByteBuffer featureImgData;
  private MappedByteBuffer featureTfLiteModel;
  private Interpreter.Options featureTfLiteOptions;
  private Interpreter featureTfLite;

  private Bitmap testBitmap;
  private static final boolean FEATURE_MODEL_TEST = false;


  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized,
      final boolean includeKeypoints,
      final String featureModelFilename,
      final String featuresFilename,
      final float minimumConfidence)
      throws IOException {
    if (!OpenCVLoader.initDebug()) {
      OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, context, null);
      Log.e("OpenCv", "Unable to load OpenCV");
    }
    else {
      Log.d("OpenCv", "OpenCV loaded");
    }

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
    MetadataExtractor metadata = new MetadataExtractor(modelFile);
    InputStream is = context.getAssets().open(labelFilename);
    try (BufferedReader br =
                 new BufferedReader(
                         new InputStreamReader(is))) {
      String line;
      while ((line = br.readLine()) != null) {
        Log.w(TAG, line);
        d.labels.add(line);
      }
    }
//    try (BufferedReader br =
//        new BufferedReader(
//            new InputStreamReader(
//                metadata.getAssociatedFile(labelFilename), Charset.defaultCharset()))) {
//      String line;
//      while ((line = br.readLine()) != null) {
//        Log.w(TAG, line);
//        d.labels.add(line);
//      }
//    }

    d.inputSize = inputSize;

    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      options.setUseXNNPACK(true);
      // options.addDelegate(new GpuDelegate());
      // options.addDelegate(new NnApiDelegate());
      //d.gpuDelegate = new GpuDelegate();
      //options.addDelegate(d.gpuDelegate);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    d.includeKeypoints = includeKeypoints;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    if (includeKeypoints){
      d.detectMinConfidence = minimumConfidence;
      InputStream featuresIs = context.getAssets().open(featuresFilename);
      try (BufferedReader featuresBr =
                   new BufferedReader(
                           new InputStreamReader(featuresIs))) {
        String featuresLine;
        while ((featuresLine = featuresBr.readLine()) != null) {
          Log.w(TAG, featuresLine);
          int splitIndex = featuresLine.indexOf(" ");
          d.featureLabels.add(featuresLine.substring(0, splitIndex));
          List<String> featureStrList = Arrays.asList(featuresLine.substring(splitIndex+1).split(","));
          float[] feature = new float[FEATURE_MODEL_FEATURE_SIZE];
          for (int f=0;f<FEATURE_MODEL_FEATURE_SIZE;f++)
            feature[f] = Float.valueOf(featureStrList.get(f));
          d.featureValues.add(feature);
        }
      }
      MappedByteBuffer featureModelFile = loadModelFile(context.getAssets(), featureModelFilename);
      try {
        Interpreter.Options featureOptions = new Interpreter.Options();
        featureOptions.setNumThreads(1);
        // cpu
        featureOptions.setUseXNNPACK(true);
        // gpu
        // d.featureGpuDelegate = new GpuDelegate();
        // featureOptions.addDelegate(d.featureGpuDelegate);
        d.featureTfLite = new Interpreter(featureModelFile, featureOptions);
        d.featureTfLiteModel = featureModelFile;
        d.featureTfLiteOptions = featureOptions;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      d.featureImgData = ByteBuffer.allocateDirect(1 * FEATURE_MODEL_INPUT_SIZE * FEATURE_MODEL_INPUT_SIZE * 3 * 4);
      d.featureImgData.order(ByteOrder.nativeOrder());
      d.featureMatIntValues = new byte[FEATURE_MODEL_INPUT_SIZE * FEATURE_MODEL_INPUT_SIZE * 4];

      d.outputKeypoints = new float[1][NUM_DETECTIONS][4][2];
      d.outputKeypointScores = new float[1][NUM_DETECTIONS][4];
      d.outputFeatureScores = new float[1][FEATURE_MODEL_CLASS_NUM];
      d.outputFeatures = new float[1][FEATURE_MODEL_FEATURE_SIZE];

      if (d.FEATURE_MODEL_TEST){
        try
        {
          InputStream imageis = context.getAssets().open("test.jpg");
          d.testBitmap = BitmapFactory.decodeStream(imageis);
          is.close();
        }
        catch (IOException e)
        {
          e.printStackTrace();
        }
      }
    }
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(
          final Bitmap bitmap,
          final Bitmap srcBitmap
  ) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();

    if (isModelQuantized){
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = intValues[i * inputSize + j];
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        }
      }
    }else if(includeKeypoints){
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = intValues[i * inputSize + j];
          imgData.putFloat(((pixelValue >> 16) & 0xFF));
          imgData.putFloat(((pixelValue >> 8) & 0xFF));
          imgData.putFloat((pixelValue & 0xFF));
        }
      }
    }else{
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = intValues[i * inputSize + j];
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];
    if (includeKeypoints){
      outputKeypoints = new float[1][NUM_DETECTIONS][4][2];
      outputKeypointScores = new float[1][NUM_DETECTIONS][4];
    }

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    if (includeKeypoints){
      outputMap.put(4, outputKeypoints);
      outputMap.put(5, outputKeypointScores);
    }
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.
    // You need to use the number of detections from the output and not the NUM_DETECTONS variable
    // declared on top
    // because on some models, they don't always output the same total number of detections
    // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
    // If you don't use the output's numDetections, you'll get nonsensical data
    int numDetectionsOutput =
        min(
            NUM_DETECTIONS,
            (int) numDetections[0]); // cast from float to integer, use min for safety

    if (includeKeypoints && numDetectionsOutput>0){
      numDetectionsOutput = outputScores[0][0] >= detectMinConfidence ? 1 : 0;
    }
    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    for (int i = 0; i < numDetectionsOutput; ++i) {
      final RectF detection =
          new RectF(
              outputLocations[0][i][1] * inputSize,
              outputLocations[0][i][0] * inputSize,
              outputLocations[0][i][3] * inputSize,
              outputLocations[0][i][2] * inputSize);
      if (includeKeypoints){
        String featureLableStr = extractFeature(srcBitmap, outputKeypoints[0][i]);
        recognitions.add(new Recognition("" + i, featureLableStr, outputScores[0][i], detection));
        break;
      }
      recognitions.add(
          new Recognition(
              "" + i, labels.get((int) outputClasses[0][i]), outputScores[0][i], detection));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  private String extractFeature(
          Bitmap bitmap,
          float[][] keypoints
  ){
    int wh_max = Math.max(bitmap.getWidth(), bitmap.getHeight());

    List<Point> srcPoints = new ArrayList<>();
    for(int i=0;i<4;i++){
      srcPoints.add(new Point(wh_max*keypoints[i][1], wh_max*keypoints[i][0]));
    }
    Mat src = Converters.vector_Point2f_to_Mat(srcPoints);

    List<Point> dstPoints = new ArrayList<>();
    dstPoints.add(new Point(0, 24));
    dstPoints.add(new Point(224, 24));
    dstPoints.add(new Point(212, 224));
    dstPoints.add(new Point(12, 224));
    Mat dst = Converters.vector_Point2f_to_Mat(dstPoints);

    Mat m = Imgproc.getPerspectiveTransform(src, dst);
    Mat srcMat = new Mat();
    Utils.bitmapToMat(bitmap, srcMat);

    Mat dstMat = new Mat();
    if (FEATURE_MODEL_TEST){
      saveBitmap(bitmap, "srcMat.png");
      Utils.bitmapToMat(testBitmap, dstMat);
      Bitmap tmp = Bitmap.createBitmap(dstMat.cols(), dstMat.rows(),
              Bitmap.Config.ARGB_8888);
      Utils.matToBitmap(dstMat, tmp);
      saveBitmap(tmp, "aligned.png");
    }else{
      Imgproc.warpPerspective(
              srcMat,
              dstMat,
              m,
              new Size(FEATURE_MODEL_INPUT_SIZE, FEATURE_MODEL_INPUT_SIZE)
      );
    }

    dstMat.get(0, 0, featureMatIntValues);

    featureImgData.rewind();
    for (int i = 0; i < FEATURE_MODEL_INPUT_SIZE; ++i) {
      for (int j = 0; j < FEATURE_MODEL_INPUT_SIZE; ++j) {
        int pixelIndex = (i * FEATURE_MODEL_INPUT_SIZE + j) * 4;
        featureImgData.putFloat((featureMatIntValues[pixelIndex+0]&0xFF)/255.0f);
        featureImgData.putFloat((featureMatIntValues[pixelIndex+1]&0xFF)/255.0f);
        featureImgData.putFloat((featureMatIntValues[pixelIndex+2]&0xFF)/255.0f);
      }
    }
    Object[] inputArray = {featureImgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputFeatureScores);
    outputMap.put(1, outputFeatures);
    featureTfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    int minDistanceIndex = -1;
    double minDistance = 100;
    for(int i=0;i<featureLabels.size();i++){
      double distance = getFeatureDistance(outputFeatures[0], featureValues.get(i));
      if (distance < FEATURE_MIN_DISTANCE && distance < minDistance) {
        minDistance = distance;
        minDistanceIndex = i;
      }
    }
    if (minDistanceIndex>=0)
      return String.format("%s %.2f", featureLabels.get(minDistanceIndex), minDistance);
    else
      return "other 2.00";
  }

  private double getFeatureDistance(float[] feature1, float[] feature2){
    //np.sqrt(np.sum(np.square(output1 - output2), axis=-1))
    double distanceSum = 0;
    for (int i=0;i<FEATURE_MODEL_FEATURE_SIZE;i++){
      distanceSum += Math.pow(feature1[i]-feature2[i], 2);
    }
    return Math.sqrt(distanceSum);
  }


  public static void saveBitmap(final Bitmap bitmap, final String filename) {
    final String root =
            Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow";
//    Log.i("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root);
    final File myDir = new File(root);

    if (!myDir.mkdirs()) {
//      Log.i("Make dir failed");
    }

    final String fname = filename;
    final File file = new File(myDir, fname);
    if (file.exists()) {
      file.delete();
    }
    try {
      final FileOutputStream out = new FileOutputStream(file);
      bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
      out.flush();
      out.close();
    } catch (final Exception e) {
//      Log.e(e, "Exception!");
    }
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
    /*if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }*/
    if (includeKeypoints){
      if (featureTfLite != null){
        featureTfLite.close();
        featureTfLite = null;
      }
      /*if (featureGpuDelegate != null) {
        featureGpuDelegate.close();
        featureGpuDelegate = null;
      }*/
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (tfLite != null) {
      tfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
    if (includeKeypoints && featureTfLite != null){
      featureTfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) {
      tfLiteOptions.setUseNNAPI(isChecked);
      recreateInterpreter();
    }
    if (includeKeypoints && featureTfLite != null){
      featureTfLiteOptions.setUseNNAPI(isChecked);
      recreateInterpreter();
    }
  }

  private void recreateInterpreter() {
    tfLite.close();
    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
    if (includeKeypoints){
      featureTfLite.close();
      featureTfLite = new Interpreter(featureTfLiteModel, featureTfLiteOptions);
    }
  }
}
