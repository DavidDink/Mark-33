package main;

public class Container {
	public static final float DEFAULT_OUTSIDE_TEMP = 40f;
	public static final float DEFAULT_OUTSIDE_HUMIDITY = 35f;
	
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
	
	public float getInsideFeelTemp() {
		final float humDiff = ComfortManager.IDEAL_HUMIDITY - insideHumidity;
		return insideTemp - humDiff/3f;
	}

	public float insideTemp() {
		return insideTemp;
	}

	public void setInsideTemp(float insideTemp) {
		this.insideTemp = insideTemp;
	}
	
	public void adjustInsideTemp(float adj) {
		setInsideTemp(insideTemp + adj);
	}

	public float insideHumidity() {
		return insideHumidity;
	}

	public void setInsideHumidity(float insideHumidity) {
		this.insideHumidity = insideHumidity;
	}
	
	public void adjustInsideHumidity(float adj) {
		setInsideHumidity(insideHumidity + adj);
	}

	public float outsideTemp() {
		return outsideTemp;
	}

	public void setOutsideTemp(float outsideTemp) {
		this.outsideTemp = outsideTemp;
	}
	
	public void adjustOutsideTemp(float adj) {
		setOutsideTemp(outsideTemp + adj);
	}

	public float outsideHumidity() {
		return outsideHumidity;
	}

	public void setOutsideHumidity(float outsideHumidity) {
		this.outsideHumidity = outsideHumidity;
	}
	
	public void adjustOutsideHumidity(float adj) {
		setOutsideHumidity(outsideHumidity + adj);
	}
}
