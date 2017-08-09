package main;

public class Container {
	public static final float DEFAULT_INSIDE_TEMP = 60f;
	public static final float DEFAULT_OUTSIDE_TEMP = 40f;
	
	private float insideTemp;
	private float insideHumidity;
	private float outsideTemp;
	private float outsideHumidity;
	
	public Container(float insideTemp, float outsideTemp, float insideHumidity, float outsideHumidity) {
		this.insideTemp = insideTemp;
		this.insideHumidity = insideHumidity;
		this.outsideTemp = outsideTemp;
		this.outsideHumidity = outsideHumidity;
	}
	
	public Container(float insideTemp, float outsideTemp) {
		this(insideTemp, outsideTemp, 0f, 0f);
	}
	
	public Container() {
		this(DEFAULT_INSIDE_TEMP, DEFAULT_OUTSIDE_TEMP);
	}
	
	public Container(Container copy) {
		this(copy.insideTemp, copy.outsideTemp, copy.insideHumidity, copy.outsideHumidity);
	}

	public float getInsideTemp() {
		return insideTemp;
	}

	public void setInsideTemp(float insideTemp) {
		this.insideTemp = insideTemp;
	}

	public float getInsideHumidity() {
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
