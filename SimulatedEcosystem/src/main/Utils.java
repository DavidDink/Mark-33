package main;

public final class Utils {
	private Utils() {}
	
	public static boolean withinRange(float val, float desiredVal, float acceptableOffset) {
		return Math.abs(desiredVal - val) <= acceptableOffset;
	}
}
