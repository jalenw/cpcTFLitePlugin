/********* TensorflowPlugin.m Cordova Plugin Implementation *******/

#import <Cordova/CDV.h>
#import "OpenCVWrapper.h"
@interface cpcTFLitePlugin : CDVPlugin {
  // Member variables go here.
}

- (void)coolMethod:(CDVInvokedUrlCommand*)command;
@end

@implementation cpcTFLitePlugin

- (void)coolMethod:(CDVInvokedUrlCommand*)command
{
//    CDVPluginResult* pluginResult = nil;
    NSString* echo = [command.arguments objectAtIndex:0];
    [OpenCVWrapper saveArguments:echo];
//
//    if (echo != nil && [echo length] > 0) {
//        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:echo];
//    } else {
//        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_ERROR];
//    }
//
//    [self.commandDelegate sendPluginResult:pluginResult callbackId:command.callbackId];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (uint64_t) NSEC_PER_SEC), dispatch_get_main_queue(), CFBridgingRelease(CFBridgingRetain(^(void) {
        UIStoryboard *mainStoryBoard = [UIStoryboard storyboardWithName:@"TensorflowSB"bundle:nil];
        UIViewController *secondViewController = [mainStoryBoard instantiateViewControllerWithIdentifier:@"ViewController"];
        [((CDVAppDelegate*)[[UIApplication sharedApplication] delegate]).window.rootViewController presentViewController:secondViewController animated:YES completion:^{
            
        }];
    })));
}
@end
