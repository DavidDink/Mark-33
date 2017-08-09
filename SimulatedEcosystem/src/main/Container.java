package main;

public class Container {
	public static final float DEFAULT_OUTSIDE_TEMP = 40f;
	public static final float DEFAULT_OUTSIDE_HUMIDITY = 55f;
	
	private float insideTemp;
	private float insideHumidity;
	private float outsideTemp;
	private float outsideHumidity;
	
	public Container(float insideTemp, float outsideTemp,
			float insideHumidity, float outsideHumidity) {
		this.insideTemp = insideTemp;
		this.insideHumidity = insideHumidity;
		this.outsideTemp = outsideTemp;
		this.outsideHumidity = outsideHumidity;
	}
	
	public Container(float insideTemp, float outsideTemp) {
		this(insideTemp, outsideTemp, ComfortManager.IDEAL_HUMIDITY,
				DEFAULT_OUTSIDE_HUMIDITY);
	}
	
	public Container() {
		this(ComfortManager.IDEAL_TEMP, DEFAULT_OUTSIDE_TEMP);
	}
	
	public Container(Container copy) {
		this(copy.insideTemp, copy.outsideTemp, copy.insideHumidity,
				copy.outsideHumidity);
	}

	public float insideTemp() {
		return insideTemp;
	}

	public void setInsideTemp(float insideTemp) {
		this.insideTemp = insideTemp;
	}

	public float insideHumidity() {
		return insideHumidity;
	}

	public void setInsideHumidity(float insideHumidity) {
		this.insideHumidity = insideHumidity;
	}

	public float getOutsideTemp() {
		return outsideTemp;
	}

	public void setOutsideTemp(float outsideTemp) {
		this.outsideTemp = outsideTemp;
	}

	public float getOutsideHumidity() {
		return outsideHumidity;
	}

	public void setOutsideHumidity(float outsideHumidity) {
		this.outsideHumidity = outsideHumidity;
	}
}
