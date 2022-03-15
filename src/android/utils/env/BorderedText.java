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

package com.cpc.cpctfliteandroid.utils.env;
//package org.tensorflow.lite.examples.detection.env;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Align;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.Typeface;
import java.util.Vector;
import android.text.Layout;
import android.text.TextPaint;
import android.text.StaticLayout;

/** A class that encapsulates the tedious bits of rendering legible, bordered text onto a canvas. */
public class BorderedText {
  private final Paint interiorPaint;
  private final Paint exteriorPaint;

  private final float textSize;

  /**
   * Creates a left-aligned bordered text object with a white interior, and a black exterior with
   * the specified text size.
   *
   * @param textSize text size in pixels
   */
  public BorderedText(final float textSize) {
    this(Color.WHITE, Color.BLACK, textSize);
  }

  /**
   * Create a bordered text object with the specified interior and exterior colors, text size and
   * alignment.
   *
   * @param interiorColor the interior text color
   * @param exteriorColor the exterior text color
   * @param textSize text size in pixels
   */
  public BorderedText(final int interiorColor, final int exteriorColor, final float textSize) {
    interiorPaint = new Paint();
    interiorPaint.setTextSize(textSize);
    interiorPaint.setColor(interiorColor);
    interiorPaint.setStyle(Style.FILL);
    interiorPaint.setAntiAlias(false);
    interiorPaint.setAlpha(255);

    exteriorPaint = new Paint();
    exteriorPaint.setTextSize(textSize);
    exteriorPaint.setColor(exteriorColor);
    exteriorPaint.setStyle(Style.FILL_AND_STROKE);
    exteriorPaint.setStrokeWidth(textSize / 8);
    exteriorPaint.setAntiAlias(false);
    exteriorPaint.setAlpha(255);

    this.textSize = textSize;
  }

  public void setTypeface(Typeface typeface) {
    interiorPaint.setTypeface(typeface);
    exteriorPaint.setTypeface(typeface);
  }

  public void drawText(final Canvas canvas, final float posX, final float posY, final String text) {
    canvas.drawText(text, posX, posY, exteriorPaint);
    canvas.drawText(text, posX, posY, interiorPaint);
  }

  public void drawText(final Canvas canvas, final float posX, final float posY, final String text, Paint bgPaint) {
    float width = exteriorPaint.measureText(text);
    float textSize = exteriorPaint.getTextSize();
    Paint paint = new Paint(bgPaint);
    paint.setStyle(Paint.Style.FILL);
    paint.setAlpha(160);
    //canvas.drawRect(posX, (posY + (int) (textSize)), (posX + (int) (width)), posY, paint);
    int line = (text.length() - text.replaceAll(";","").length()) + 1;

    String[] textArray = text.split(";");
    int index = 0;
    int elementLength = textArray[0].length();
    for(int i=1; i< textArray.length; i++) {
      if(textArray[i].length() > elementLength) {
        index = i; elementLength = textArray[i].length();
      }
    }
    width = exteriorPaint.measureText(textArray[index]);

    canvas.drawRect(posX, ((posY + (int) (textSize) * line)), (posX + (int) (width)), posY, paint);

    //canvas.drawText(text, posX, (posY + textSize), interiorPaint);

    TextPaint mTextPaint=new TextPaint();
    mTextPaint.setTextSize(50);
    mTextPaint.setColor(Color.WHITE);
    StaticLayout mTextLayout = new StaticLayout(text.replace(";", " or\n"), mTextPaint, canvas.getWidth(), Layout.Alignment.ALIGN_NORMAL, 1.0f, 0.0f, false);

    canvas.save();

    Float textX = posX;
    Float textY = posY;

    canvas.translate(textX, textY);
    mTextLayout.draw(canvas);
    canvas.restore();
  }

  public void drawLines(Canvas canvas, final float posX, final float posY, Vector<String> lines) {
    int lineNum = 0;
    for (final String line : lines) {
      drawText(canvas, posX, posY - getTextSize() * (lines.size() - lineNum - 1), line);
      ++lineNum;
    }
  }

  public void setInteriorColor(final int color) {
    interiorPaint.setColor(color);
  }

  public void setExteriorColor(final int color) {
    exteriorPaint.setColor(color);
  }

  public float getTextSize() {
    return textSize;
  }

  public void setAlpha(final int alpha) {
    interiorPaint.setAlpha(alpha);
    exteriorPaint.setAlpha(alpha);
  }

  public void getTextBounds(
      final String line, final int index, final int count, final Rect lineBounds) {
    interiorPaint.getTextBounds(line, index, count, lineBounds);
  }

  public void setTextAlign(final Align align) {
    interiorPaint.setTextAlign(align);
    exteriorPaint.setTextAlign(align);
  }
}
