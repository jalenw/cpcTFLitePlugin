console.log("Running hook to install nodes modules pre-requisites");

module.exports = function (context) {
  var child_process = require('child_process');
      //deferral = context.requireCordovaModule('q').defer();

  var output = child_process.exec('npm install', {cwd: __dirname}, function (error) {
    if (error !== null) {
      console.log('exec error: ' + error);
      //deferral.reject('npm installation failed');
    }
    else {
      //deferral.resolve();
      console.log('ok.....' + error);
      var add_swift = require('./add-swift-support.js');
      add_swift(context);	    
    }
  });

  //return deferral.promise;
};
