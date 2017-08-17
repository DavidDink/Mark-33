package main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * PLEASE READ: CURRENTLY THIS CLASS WILL ONLY
 * FUNCTION PROPERLY IF THE "intervalsPerDay" 
 * PARAMETER IS SET TO 24. ANY OTHER VALUE
 * MAY RESULT IN UNDESIRED BEHAVIOR.
 * @author David Dinkevich
 */
public class Day {
	private static final int SECONDS_PER_DAY = 86400;
	// Each Day is divided into 8 "parts"
	private static final int PARTS_PER_DAY = 8;
	// How many hours are in each part
	private static final int HOURS_PER_PART = 3;
	
	private int intervalsPerDay;
	private List<Interval> intervals;
	
	// "The temperature today will be a high of 85 and a low of 70..."
	private float highTemp, lowTemp;
	// The initial humidity. All following humidity readings will randomly deviate
	// from this initial value
	private float initHum;
	
	public Day(int intervalsPerDay, float lowTemp, float highTemp, float initialHum) {
		this.intervalsPerDay = Math.abs(intervalsPerDay);
		this.highTemp = highTemp;
		this.lowTemp = lowTemp;
		initHum = initialHum;
	}
	
	public Day(int intervalsPerDay) {
		// Default high and low temps, initial humidity
		this(intervalsPerDay, 70f, 85f, ComfortManager.IDEAL_HUMIDITY);
	}
	
	public Day(float lowTemp, float highTemp, float initialHum) {
		this(24, lowTemp, highTemp, initialHum);
	}
	
	public Day() {
		this(24);
	}
	
	public static Day randomDay(int numIntervals) {
		// Max temp
		final float highTemp = 20 + Utils.RANDOM.nextFloat() * 80f;
		// Min temp
		final float lowTemp = highTemp - 20f;
		final float initialHum = 10f + Utils.RANDOM.nextFloat() * 70f;
		return new Day(numIntervals, lowTemp, highTemp, initialHum);
	}
	
	public static Day randomDay() {
		return randomDay(24);
	}
	
	// Seconds = 0 to 86400
	private int getPartThatContains(float seconds) {
		final float hours = Math.abs(seconds) / 60f / 60f;
		return (int)(hours - (hours % HOURS_PER_PART)) / HOURS_PER_PART;
	}
	
	private float[] generateTemperatures() {
		float[] temps = new float[intervalsPerDay];
		
		// The part of day that is the hottest (noon).
		// The plus one is because we want the part after which the
		// temperature will begin to get cooler
		final int peakPart = PARTS_PER_DAY/2;
		// The amount the temp will change every "part"
		float dTemp = (highTemp - lowTemp) / (PARTS_PER_DAY/2);
						
		for (int i = 0; i < temps.length; i++) {
			final int seconds = i * 60 * 60; // Hours to seconds
			final int partInDay = getPartThatContains(seconds);
			
			if (partInDay == peakPart)
				temps[i] = highTemp;
			else if (partInDay < peakPart)
				temps[i] = lowTemp + (partInDay * dTemp);
			else
				temps[i] = highTemp - ((partInDay-peakPart) * dTemp);
			// Add a little variation
			temps[i] += Utils.standardRandomFloat();
		}
		
		return temps;
	}
	
	public float[] generateHumidities() {
		float[] humidities = new float[intervalsPerDay];
		// Set initial temp
		humidities[0] = initHum;
		
		for (int i = 1; i < humidities.length; i++) {
			// Magnitude of change in humidity
			final float mag = 2f;
			// Change in humidity
			final float dHumidity = Utils.standardRandomFloat() * mag;
			// Humidity must be >= 0
			final float newVal = Math.max(0f, humidities[i-1] + dHumidity);
			// Humidity must be <= 100
			humidities[i] = Math.min(newVal, 100f);
		}
		
		return humidities;
	}
	
	public List<Interval> getIntervals() {
		if (intervals == null) {
			intervals = new ArrayList<>(intervalsPerDay);
			float[] temps = generateTemperatures();
			float[] humidities = generateHumidities();
			for (int i = 0; i < intervalsPerDay; i++) {
				// Insert the timestamp, outside temp, and outside humidity
				Interval interval = new Interval();
				interval.setTimestamp(i+1);
				interval.setOutsideTemp(temps[i]);
				interval.setOutsideHum(humidities[i]);
				intervals.add(interval);
			}
		}
		return Collections.unmodifiableList(intervals);
	}
	
	public int getIntervalsPerDay() {
		return intervalsPerDay;
	}
	
	public int getIntervalLength() {
		return SECONDS_PER_DAY / intervalsPerDay;
	}	
	
	public float getHighTemp() {
		return highTemp;
	}
	
	public void setHighTemp(float highTemp) {
		this.highTemp = highTemp;
	}
	
	public float getLowTemp() {
		return lowTemp;
	}
	
	public void setLowTemp(float lowTemp) {
		this.lowTemp = lowTemp;
	}
	
//	public static enum Weather {
//		REGULAR(80f, 68f, 40f), RAIN(68f, 50f, 67f), SNOW(), HEAT_WAVE, CLOUDY;
//		
//		public final float MAX_TEMP, MIN_TEMP, INIT_HUMIDITY;
//		
//		private Weather(float maxTemp, float minTemp, float initHum) {
//			MAX_TEMP = maxTemp;
//			MIN_TEMP = minTemp;
//			INIT_HUMIDITY = initHum;
//		}
//	}
}
