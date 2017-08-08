package main;

public class Container {
	private static final float DEFAULT_INSIDE_TEMP = 60f;
	
	private float insideTemp;
	private float insideHumidity;
	private float outsideTemp;
	private float outsideHumidity;
	
	public Container(float insideTemp, float insideHumidity, float outsideTemp, float outsideHumidity) {
		this.insideTemp = insideTemp;
		this.insideHumidity = insideHumidity;
		this.outsideTemp = outsideTemp;
		this.outsideHumidity = outsideHumidity;
	}
	
	public Container(float insideTemp, float insideHumidity) {
		this(insideTemp, insideHumidity, 0f, 0f);
	}
	
	public Container() {
		this(DEFAULT_INSIDE_TEMP, 50);
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
