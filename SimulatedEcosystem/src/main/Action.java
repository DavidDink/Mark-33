package main;

public class Action {
	public static enum Types {
		AC_ON("ac on"), AC_OFF("ac off"), HEATER_ON("heater on"),
		HEATER_OFF("heater off"), AC_ON_HEATER_ON("ac on - heater on"),
		AC_ON_HEATER_OFF("ac on - heater off"), AC_OFF_HEATER_ON("ac off - heater on"),
		AC_OFF_HEATER_OFF("ac off - heater off"), NONE("no action");
		
		private String name;
		
		private Types(String name) {
			this.name = name;
		}
		
		@Override
		public String toString() {
			return name;
		}
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
	
	@Override
	public String toString() {
		return type.toString();
	}
	
	public float getCost() {
		return powerUsed;
	}

	public Types getType() {
		return type;
	}

	public float getPowerUsed() {
		return powerUsed;
	}
	
}
