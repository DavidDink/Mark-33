package main;

public class Action {
	public static enum Types {
		AC_ON, AC_OFF, HEATER_ON, HEATER_OFF, NONE;
	}
	
	private Types type;
	private float powerUsed;
	
	public Action(Types type, float powerUsed) {
		this.type = type;
		this.powerUsed = powerUsed;
	}
	
	public Action() {
		this(Types.NONE, 0);
	}
	
	public float getCost() {
		return Math.abs(powerUsed);
	}

	public Types getType() {
		return type;
	}

	public float getPowerUsed() {
		return powerUsed;
	}
	
}
