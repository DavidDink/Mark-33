package main;

public class TempChanger {
	private float minPower, maxPower;
	private float power;
	
	public TempChanger(float minPower, float maxPower) {
		this.minPower = minPower;
		this.maxPower = maxPower;
	}
	
	public TempChanger(TempChanger other) {
		this(other.minPower, other.maxPower);
		power = other.power;
	}
	
	public TempChanger() {
		this(-2f, 2f);
	}
	
	public void setPower(float power) {
		this.power = power > maxPower ? maxPower : power < minPower ? minPower : power;
	}
	
	public float getPower() {
		return power;
	}
	
	public float getMinPower() {
		return minPower;
	}
	
	public void setMinPower(float minPower) {
		this.minPower = minPower;
	}
	
	public float getMaxPower() {
		return maxPower;
	}
	
	public void setMaxPower(float maxPower) {
		this.maxPower = maxPower;
	}
}
