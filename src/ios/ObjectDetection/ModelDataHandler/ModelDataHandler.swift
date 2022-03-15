// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
    let cvPixelBuff : CVPixelBuffer
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
  let outputKeypoint: ArraySlice<Float>
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum MobileNetSSD {
  static let modelInfo: FileInfo = (name: "centernet", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labelmap_centernet", extension: "txt")
    static let featuresModelInfo: FileInfo = (name: "model", extension: "tflite")
    static let featuresLabelsInfo: FileInfo = (name: "features", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  let threadCountLimit = 10

  let threshold: Float = 0.1

  // MARK: Model parameters
  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 256
  let inputHeight = 256

  let featuresInputWidth = 224
  let featuresInputHeight = 224
    
  let FEATURE_MODEL_FEATURE_SIZE = 128
  let FEATURE_MIN_DISTANCE = 1.2
    
  // image mean and std for floating model, should be consistent with parameters used in model training
  let imageMean: Float = 0
  let imageStd:  Float = 1

  // MARK: Private properties
  private var labels: [String] = []
    private var featuresLabels: [String] = []
    private var featuresValuesLabels: [[String]] = []
    
  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter
    private var featuresInterpreter: Interpreter
    
  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3
  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
      
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
      
      let featuresModelFilename = MobileNetSSD.featuresModelInfo.name

      // Construct the path to the model file.
      guard let featuresModelPath = Bundle.main.path(
        forResource: featuresModelFilename,
        ofType: MobileNetSSD.featuresModelInfo.extension
      ) else {
        print("Failed to load the model file with name: \(featuresModelFilename).")
        return nil
      }
      
      var featuresOptions = Interpreter.Options()
      featuresOptions.threadCount = threadCount
      do {
        // Create the `Interpreter`.
        featuresInterpreter = try Interpreter(modelPath: featuresModelPath, options: featuresOptions)
        // Allocate memory for the model's input `Tensor`s.
        try featuresInterpreter.allocateTensors()
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        return nil
      }

    super.init()

    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
      
    loadFeaturesLabels(fileInfo: MobileNetSSD.featuresLabelsInfo)
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    
//    let rect = CGRect.init(origin: CGPoint.init(x: 0, y: (imageHeight-imageWidth)/2), size: CGSize.init(width: imageWidth, height: imageWidth))
      
//    let pixelBuffer = PixelBufferCut.createBuffer(pixelBuffer, rect: rect).takeUnretainedValue()
      
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)

    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }

    let interval: TimeInterval
    let outputBoundingBox: Tensor
    let outputClasses: Tensor
    let outputScores: Tensor
    let outputCount: Tensor
      let outputKeypoints: Tensor
      let outputKeypointScores: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8,
        foFeaturesModel: false
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)
        
      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000

      outputBoundingBox = try interpreter.output(at: 0)
      outputClasses = try interpreter.output(at: 1)
      outputScores = try interpreter.output(at: 2)
      outputCount = try interpreter.output(at: 3)
        outputKeypoints = try interpreter.output(at: 4)
        outputKeypointScores = try interpreter.output(at: 5)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Formats the results
    let resultArray = formatResults(
      boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
      outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
      outputScores: [Float](unsafeData: outputScores.data) ?? [],
      outputCount: Int(([Float](unsafeData: outputCount.data) ?? [0])[0]),
      outputKeypoints: [Float](unsafeData: outputKeypoints.data) ?? [],
      outputKeypointScores: [Float](unsafeData: outputKeypointScores.data) ?? [],
      width: CGFloat(imageWidth),
      height: CGFloat(imageHeight)
    )

    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: resultArray, cvPixelBuff: scaledPixelBuffer)
    return result
  }

    var tempIndex: Int = 0
    var tempScore: Float = 0.00
  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
  func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, outputKeypoints: [Float],outputKeypointScores: [Float],width: CGFloat, height: CGFloat) -> [Inference]{
      var resultsArray: [Inference] = []
      if (outputCount == 0) {
          return resultsArray
      }

      tempIndex = 0
      tempScore = 0.00
      for i in 0...outputCount - 1 {
          
          let score = outputScores[i]
          
          // Filters results with confidence < threshold.
          guard score >= threshold else {
              continue
          }
          
          if(score>tempScore){
              tempScore = score;
              tempIndex = i;
          }
      }
      
      if(tempScore>=threshold){
          // Gets the output class names for detected classes from labels list.
          let outputClassIndex = Int(outputClasses[tempIndex])
          let outputClass = labels[outputClassIndex]
          let outputKeypoint = outputKeypoints[8*tempIndex...8*tempIndex+7]
          
          var rect: CGRect = CGRect.zero
          
          // Translates the detected bounding box to CGRect.
          rect.origin.y = CGFloat(boundingBox[4*tempIndex])
          rect.origin.x = CGFloat(boundingBox[4*tempIndex+1])
          rect.size.height = CGFloat(boundingBox[4*tempIndex+2]) - rect.origin.y
          rect.size.width = CGFloat(boundingBox[4*tempIndex+3]) - rect.origin.x
          
          // The detected corners are for model dimensions. So we scale the rect with respect to the
          // actual image dimensions.
          let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
          
          // Gets the color assigned for the class
          let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
          let inference = Inference(confidence: tempScore,
                                    className: outputClass,
                                    rect: newRect,
                                    displayColor: colorToAssign,
                                    outputKeypoint: outputKeypoint)
          resultsArray.append(inference)
      }

    return resultsArray
  }
    
    func recognizeImage(result:Result) -> String {
        var name:String = ""
        
        let cvPixelBuff = result.cvPixelBuff
        
        var outputKeypoint:[Float] = []
        if(result.inferences.count==0){
            return ""
        }else {
            var kps = result.inferences[0].outputKeypoint
            for i in 0..<kps.count {
                kps[i] = kps[i]*Float(inputWidth)
            }
            outputKeypoint = Array(kps)
        }
        
        let openCVWrapper = OpenCVWrapper();
        let resultPixelBuffer = openCVWrapper.confirmedImage(cvPixelBuff, points: outputKeypoint).takeUnretainedValue()

        let scaledSize = CGSize(width: featuresInputWidth, height: featuresInputHeight)
        guard let scaledPixelBuffer = resultPixelBuffer.resized(to: scaledSize) else {
          return ""
        }

        let outputFeatureScores: Tensor
        let outputFeatures: Tensor
        do {
            let inputTensor = try featuresInterpreter.input(at: 0)

            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
              byteCount: batchSize * featuresInputWidth * featuresInputHeight * inputChannels,
              isModelQuantized: inputTensor.dataType == .uInt8,
                foFeaturesModel: true
            ) else {
              print("Failed to convert the image buffer to RGB data.")
              return ""
            }

            // Copy the RGB data to the input `Tensor`.
            try featuresInterpreter.copy(rgbData, toInputAt: 0)

            // Run inference by invoking the `Interpreter`.
            try featuresInterpreter.invoke()

            outputFeatureScores = try featuresInterpreter.output(at: 0)
            outputFeatures = try featuresInterpreter.output(at: 1)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return ""
        }
        
//        for i in 0...featuresValuesLabels.count {
//            let score = [Float](unsafeData: outputFeatures.data)?[i]
//            print("outputfeatures \(String(describing: score))")
//        }
        
        var minDistanceIndex = -1;
        var minDistance = 100.00;
        for i in 0...featuresValuesLabels.count-1{
            let distance = getFeatureDistance(feature1: [Float](unsafeData: outputFeatures.data)!, feature2: featuresValuesLabels[i]);
            if (distance < FEATURE_MIN_DISTANCE && distance < minDistance) {
                minDistance = distance;
                minDistanceIndex = i;
            }
        }
        if (minDistanceIndex>=0){
            name = featuresLabels[minDistanceIndex]
        }
        return name
    }
    
    private func getFeatureDistance(feature1:[Float],feature2:[String]) -> Double {
        var distanceSum:Double = 0;
        for i in 0...FEATURE_MODEL_FEATURE_SIZE-1{
            distanceSum += pow(Double(feature1[i]-Float(feature2[i])!), 2);
        }
        return sqrt(distanceSum);
    }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }
    
    private func loadFeaturesLabels(fileInfo: FileInfo) {
      let filename = fileInfo.name
      let fileExtension = fileInfo.extension
      guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
        fatalError("Labels file not found in bundle. Please add a labels file with name " +
                     "\(filename).\(fileExtension) and try again.")
      }
      do {
        let contents = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = contents.components(separatedBy: .newlines)
          for i in 0 ..< lines.count {
              var str = lines[i]
              if(str.count>0){
                  guard let index = str.firstIndex(of: " ") else { return }
                  let featuresName = str.substring(to: index)
                  str = str.replacingOccurrences(of: " ", with: "")
                  let array = str.substring(from: index).components(separatedBy: ",")
                  featuresLabels.append(featuresName)
                  featuresValuesLabels.append(array)
              }
          }

      } catch {
        fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                     "valid labels file and try again.")
      }
    }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool,
    foFeaturesModel:Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
      free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)
    
    if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
      vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
      return byteData
    }

    // Not quantized, convert to floats
      let bytes = Array<UInt8>(unsafeData: byteData)!
      var floats = [Float]()
      if(foFeaturesModel){
          for i in 0..<bytes.count {
              floats.append((Float(bytes[i]) - 0) / 255.0)
          }
      }else{
          for i in 0..<bytes.count {
              floats.append((Float(bytes[i]) - imageMean) / imageStd)
          }
      }
    return Data(copyingBufferOf: floats)
  }

  /// This assigns color for a particular class.
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base
    // colors to each object based on its index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
