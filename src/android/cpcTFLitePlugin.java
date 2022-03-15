package org.apache.cordova.cpctflite;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

import android.widget.Toast;

import com.cpc.cpctfliteandroid.activity.DetectorActivity;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CallbackContext;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * This class echoes a string called from JavaScript.
 */
public class cpcTFLitePlugin extends CordovaPlugin {

    @Override
    public boolean execute(String action, String args, CallbackContext callbackContext) throws JSONException {
		Context context = cordova.getActivity().getApplicationContext();
        if(action.equals("coolMethod")) {
            this.coolMethod(context, args);
            return true;
        }
        return false;
    }
	
	private void coolMethod(Context context, String args) {
        Intent intent = new Intent(context, DetectorActivity.class);
        intent.putExtra("BrandDataString", args);
        this.cordova.getActivity().startActivity(intent);
    }
	/*
    private void coolMethod(String message, CallbackContext callbackContext) {
        if (message != null && message.length() > 0) {
			Toast.makeText(cordova.getContext(), "CLICK ME!" + message, Toast.LENGTH_SHORT).show();

			Intent intent =new Intent(this.cordova.getActivity(), DetectorActivity.class);
			this.cordova.getActivity().startActivity(intent);
		
            callbackContext.success(message);
        } else {
            callbackContext.error("Expected one non-empty string argument.");
        }
    }*/
}
