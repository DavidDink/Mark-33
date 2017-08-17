package main;

public class ActionMap {
	private float temperatureAction, humidityAction;
	
	public ActionMap(float tempAction, float humAction) {
		this.temperatureAction = tempAction;
		this.humidityAction = humAction;
	}
	
	public ActionMap(ActionMap other) {
		set(other);
	}
	
	public ActionMap() {
	}

	public void set(ActionMap other) {
		this.temperatureAction = other.temperatureAction;
		this.humidityAction = other.humidityAction;
	}
	
	public float getTemperatureAction() {
		return temperatureAction;
	}

	public void setTemperatureAction(float temperatureAction) {
		this.temperatureAction = temperatureAction;
	}

	public float getHumidityAction() {
		return humidityAction;
	}

	public void setHumidityAction(float humidityAction) {
		this.humidityAction = humidityAction;
	}


}
