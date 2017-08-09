package main;

public final class ComfortManager {
	public static final float IDEAL_COMFORT = 0f;
	public static final float IDEAL_HUMIDITY = 45f;
	public static final float IDEAL_TEMP = 75f;
	public static final float ALLOWED_HUMIDITY_RANGE = 5f;
	public static final float ALLOWED_TEMP_RANGE = 3f;
	public static final float MIN_COMFORTABLE_TEMP = IDEAL_TEMP - ALLOWED_TEMP_RANGE;
	public static final float MAX_COMFORTABLE_TEMP = IDEAL_TEMP + ALLOWED_TEMP_RANGE;
	public static final float MIN_COMFORTABLE_HUMIDITY =
			IDEAL_HUMIDITY - ALLOWED_HUMIDITY_RANGE;
	public static final float MAX_COMFORTABLE_HUMIDITY =
			IDEAL_HUMIDITY + ALLOWED_HUMIDITY_RANGE;
		
	private ComfortManager() {
	}
	
	public static boolean isComfortableTemp(float temp) {
		return Utils.withinRange(temp, IDEAL_TEMP, ALLOWED_TEMP_RANGE);
	}
	
	public static boolean isComfortableHumidity(float humidity) {
		return Utils.withinRange(humidity, IDEAL_HUMIDITY, ALLOWED_HUMIDITY_RANGE);
	}
	
	public static float evaluateTempPenalty(float temp) {
		if (isComfortableTemp(temp))
			return 0f;
		final float distA = Math.abs(MIN_COMFORTABLE_TEMP - temp);
		final float distB = Math.abs(MAX_COMFORTABLE_TEMP - temp);
		return Math.min(distA, distB);
	}
	
	public static float evaluateHumidityPenalty(float humidity) {
		if (isComfortableHumidity(humidity))
			return 0f;
		final float distA = Math.abs(MIN_COMFORTABLE_HUMIDITY - humidity);
		final float distB = Math.abs(MAX_COMFORTABLE_HUMIDITY - humidity);
		return Math.min(distA, distB);
	}
	
	public static float evaluateComfortLevel(float temp, float humidity) {
		final float tempPenalty = evaluateTempPenalty(temp);
		final float humidityPenalty = evaluateHumidityPenalty(humidity);
		return (tempPenalty + humidityPenalty);
	}
	
	public boolean isComfortable(float val) {
		return Utils.withinRange(val, 0f, 5f);
	}
}
