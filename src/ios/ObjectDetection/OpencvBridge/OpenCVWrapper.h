//
//  NSObject+OpenCVWrapper.h
//  ObjectDetection
//
//  Created by ITS-AppTeam on 2022/3/1.
//  Copyright © 2022 Y Media Labs. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject
-(CVPixelBufferRef)confirmedImage:(CVPixelBufferRef)pixelBufferRef points:(NSArray*)arys;
-(UIImage*)imageFromPixelBuffer:(CVPixelBufferRef)pixelBufferRef;
-(CVPixelBufferRef)pixelBufferFromCGImage:(UIImage*)uiimage;
+ (void)saveArguments:(NSString*)str;
+ (NSString*)getArguments;
@end

NS_ASSUME_NONNULL_END
