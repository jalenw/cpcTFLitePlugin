//
//  NSObject+OpenCVWrapper.m
//  ObjectDetection
//
//  Created by ITS-AppTeam on 2022/3/1.
//  Copyright © 2022 Y Media Labs. All rights reserved.
//
#import "opencv2/opencv.hpp"
#import "opencv2/imgcodecs/ios.h"
#import "OpenCVWrapper.h"

@implementation OpenCVWrapper
- (CVPixelBufferRef)confirmedImage:(CVPixelBufferRef)pixelBufferRef points:(NSArray*)arys
{
    UIImage *_sourceImage = [self imageFromPixelBuffer:pixelBufferRef];
    cv::Mat originalRot = [self cvMatFromUIImage:_sourceImage];
    cv::Mat original = originalRot;
//    cv::transpose(originalRot, original);

    originalRot.release();

//    cv::flip(original, original, 1);


//    CGFloat scaleFactor = 1.0;

    CGPoint ptBottomLeft = CGPointMake([arys[7] floatValue], [arys[6] floatValue]);
    CGPoint ptBottomRight = CGPointMake([arys[5] floatValue], [arys[4] floatValue]);
    CGPoint ptTopRight = CGPointMake([arys[3] floatValue], [arys[2] floatValue]);;
    CGPoint ptTopLeft = CGPointMake([arys[1] floatValue], [arys[0] floatValue]);;

//    CGFloat w1 = sqrt(pow(ptBottomRight.x - ptBottomLeft.x , 2) + pow(ptBottomRight.x - ptBottomLeft.x, 2));
//    CGFloat w2 = sqrt(pow(ptTopRight.x - ptTopLeft.x , 2) + pow(ptTopRight.x - ptTopLeft.x, 2));
//
//    CGFloat h1 = sqrt(pow(ptTopRight.y - ptBottomRight.y , 2) + pow(ptTopRight.y - ptBottomRight.y, 2));
//    CGFloat h2 = sqrt(pow(ptTopLeft.y - ptBottomLeft.y , 2) + pow(ptTopLeft.y - ptBottomLeft.y, 2));
//
//    CGFloat maxWidth = (w1 < w2) ? w1 : w2;
//    CGFloat maxHeight = (h1 < h2) ? h1 : h2;

    cv::Point2f src[4], dst[4];
    src[0].x = ptTopLeft.x;
    src[0].y = ptTopLeft.y;
    src[1].x = ptTopRight.x;
    src[1].y = ptTopRight.y;
    src[2].x = ptBottomRight.x;
    src[2].y = ptBottomRight.y;
    src[3].x = ptBottomLeft.x;
    src[3].y = ptBottomLeft.y;

    dst[0].x = 0;
    dst[0].y = 24;
    dst[1].x = 224;
    dst[1].y = 24;
    dst[2].x = 212;
    dst[2].y = 224;
    dst[3].x = 12;
    dst[3].y = 224;

    cv::Mat undistorted = cv::Mat(cv::Size(224,224), CV_8UC1);
    cv::warpPerspective(original, undistorted, cv::getPerspectiveTransform(src, dst), cv::Size(224, 224));

    UIImage *newImage = [self UIImageFromCVMat:undistorted];

    undistorted.release();
    original.release();

    return [self pixelBufferFromCGImage:newImage];
}

- (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize() * cvMat.total()];

    CGColorSpaceRef colorSpace;

    if (cvMat.elemSize() == 1) {
     colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
     colorSpace = CGColorSpaceCreateDeviceRGB();
    }

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

    CGImageRef imageRef = CGImageCreate(cvMat.cols,          // Width
             cvMat.rows,          // Height
             8,            // Bits per component
             8 * cvMat.elemSize(),       // Bits per pixel
             cvMat.step[0],         // Bytes per row
             colorSpace,          // Colorspace
             kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault, // Bitmap info flags
             provider,          // CGDataProviderRef
             NULL,           // Decode
             false,           // Should interpolate
             kCGRenderingIntentDefault);      // Intent

    UIImage *image = [[UIImage alloc] initWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return image;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.height;
    CGFloat rows = image.size.width;

    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels

    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,     // Pointer to backing data
                cols,      // Width of bitmap
                rows,      // Height of bitmap
                8,       // Bits per component
                cvMat.step[0],    // Bytes per row
                colorSpace,     // Colorspace
                kCGImageAlphaNoneSkipLast |
                kCGBitmapByteOrderDefault); // Bitmap info flags

    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);

    return cvMat;
}


- (UIImage *)imageFromPixelBuffer:(CVPixelBufferRef)pixelBufferRef {
    CVImageBufferRef imageBuffer =  pixelBufferRef;
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    size_t bufferSize = CVPixelBufferGetDataSize(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, 0);
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, baseAddress, bufferSize, NULL);
    
    CGImageRef cgImage = CGImageCreate(width, height, 8, 32, bytesPerRow, rgbColorSpace, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrderDefault, provider, NULL, true, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(rgbColorSpace);
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    return image;
}

- (CVPixelBufferRef)pixelBufferFromCGImage:(UIImage*)uiimage{
    CGImageRef image=[uiimage CGImage];
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGBitmapContextCompatibilityKey,
                             nil];

    CVPixelBufferRef pxbuffer = NULL;

    CGFloat frameWidth = CGImageGetWidth(image);
    CGFloat frameHeight = CGImageGetHeight(image);

    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
                                          frameWidth,
                                          frameHeight,
                                          kCVPixelFormatType_32ARGB,
                                          (__bridge CFDictionaryRef) options,
                                          &pxbuffer);

    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);

    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    NSParameterAssert(pxdata != NULL);

    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();

    CGContextRef context = CGBitmapContextCreate(pxdata,
                                                 frameWidth,
                                                 frameHeight,
                                                 8,
                                                 CVPixelBufferGetBytesPerRow(pxbuffer),
                                                 rgbColorSpace,
                                                 (CGBitmapInfo)kCGImageAlphaNoneSkipFirst);
    NSParameterAssert(context);
    CGContextConcatCTM(context, CGAffineTransformIdentity);
    CGContextDrawImage(context, CGRectMake(0,
                                           0,
                                           frameWidth,
                                           frameHeight),
                       image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);

    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);

    return pxbuffer;
}

+ (void)saveArguments:(NSString*)str{
    [[NSUserDefaults standardUserDefaults] setValue:str forKey:@"arguments"];
}

+ (NSString*)getArguments{
    return [[NSUserDefaults standardUserDefaults]valueForKey:@"arguments"]==nil?@"":[[NSUserDefaults standardUserDefaults]valueForKey:@"arguments"];
}
@end
