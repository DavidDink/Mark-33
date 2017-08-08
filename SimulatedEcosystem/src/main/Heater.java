package main;

public class Heater {
	public static float DEFAULT_AIR_TEMP = 2;
	
	private float airTemp;
	
	public Heater(float airTemp) {
		this.airTemp = airTemp;
	}
	
	public Heater() {
		this(DEFAULT_AIR_TEMP);
	}

	public float getAirTemp() {
		return airTemp;
	}

	public void setAirTemp(float airTemp) {
		this.airTemp = airTemp;
	}
	
	
}
