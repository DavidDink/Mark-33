package main;

public class AirConditioner {
	public static final float DEFAULT_AIR_TEMP = 2f;
	
	private float airTemp;
	private boolean on;
	
	public AirConditioner(float airTemp) {
		this.airTemp = airTemp;
	}
	
	public AirConditioner() {
		this(DEFAULT_AIR_TEMP);
	}

	public float getAirTemp() {
		return airTemp;
	}

	public void setAirTemp(float airTemp) {
		this.airTemp = airTemp;
	}
	
	
}
