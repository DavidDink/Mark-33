package main;

public class ActionMap {
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
	
	private Types action;
	private Types state;
	private float powerUsed;
	
	public ActionMap(Types action, Types state, float powerUsed) {
		this.action = action;
		this.state = state;
		this.powerUsed = powerUsed;
	}
	
	public ActionMap() {
		this(Types.NONE, Types.NONE, 0);
	}
	
	public float getCost() {
		return powerUsed;
	}

	public Types getAction() {
		return action;
	}

	public Types getState() {
		return state;
	}

	public float getPowerUsed() {
		return powerUsed;
	}
	
}
