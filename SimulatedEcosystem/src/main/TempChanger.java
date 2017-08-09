package main;

public class TempChanger {
	private float minPower, maxPower;
	private float power;
	
	private TempChanger(float minPower, float maxPower) {
		this.minPower = minPower;
		this.maxPower = maxPower;
	}
	
	public void setPower(float power) {
		this.power = power > maxPower ? maxPower : power < minPower ? minPower : power;
	}
	
	public float getPower() {
		return power;
	}
	
	public static TempChanger newAirConditioner() {
		return new TempChanger(-2, 0);
	}
	
	public static TempChanger newHeater() {
		return new TempChanger(0, 2);
	}
}
