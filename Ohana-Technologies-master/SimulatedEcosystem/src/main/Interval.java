package main;

public class Interval {
	private int timestamp;
	private float feelTemp, insideTemp, outsideTemp;
	private float insideHum, outsideHum;
	private ActionMap action, state;
	
	public Interval(int timestamp, Container con, ActionMap state, ActionMap action) {
		this.timestamp = timestamp;
		feelTemp = con.getInsideFeelTemp();
		insideTemp = con.insideTemp();
		outsideTemp = con.outsideTemp();
		insideHum = con.insideHumidity();
		outsideHum = con.outsideHumidity();
		this.action = action;
		this.state = state;
	}
	
	public Interval(int timestamp) {
		this.timestamp = timestamp;
	}

	public Interval() {
	}
	
	public void update(Container con) {
		feelTemp = con.getInsideFeelTemp();
		insideTemp = con.insideTemp();
		outsideTemp = con.outsideTemp();
		insideHum = con.insideHumidity();
		outsideHum = con.outsideHumidity();
	}
	
	
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (!(o instanceof Interval))
			return false;
		Interval other = (Interval)o;
		return other.timestamp == timestamp;
	}
	
	@Override
	public int hashCode() {
		return timestamp;
	}
	
	@Override
	public String toString() {
		final float comfortPenalty = getComfortPenalty();
		return timestamp + "," + ComfortManager.IDEAL_TEMP + "," +
				feelTemp + "," + insideTemp + "," +
				outsideTemp + "," + ComfortManager.IDEAL_HUMIDITY +
				"," + insideHum + "," + outsideHum
				+ "," + state.getTemperatureAction() + "," +
				action.getTemperatureAction() + "," + state.getHumidityAction() + "," +
				action.getHumidityAction() + "," + comfortPenalty + "," + getCost();
	}
	
	public float getCost() {
		return Math.abs(state.getTemperatureAction()) + 
				Math.abs(state.getHumidityAction());
	}
	
	public float getComfortPenalty() {
		return ComfortManager.evaluateComfortLevel(feelTemp, insideHum);
	}
	
	public float getTotalPenalty() {
		final float comfortWeight = 2.4f;
		final float costWeight = 0.6f;
		return comfortWeight * getComfortPenalty() + getCost() * costWeight;
	}

	public int getTimestamp() {
		return timestamp;
	}
	
	public ActionMap getAction() {
		return action;
	}
	
	public ActionMap getState() {
		return state;
	}

	public float getFeelTemp() {
		return feelTemp;
	}

	public float getInsideTemp() {
		return insideTemp;
	}

	public float getOutsideTemp() {
		return outsideTemp;
	}

	public float getInsideHum() {
		return insideHum;
	}

	public float getOutsideHum() {
		return outsideHum;
	}

	public void setTimestamp(int timestamp) {
		this.timestamp = timestamp;
	}

	public void setFeelTemp(float feelTemp) {
		this.feelTemp = feelTemp;
	}

	public void setInsideTemp(float insideTemp) {
		this.insideTemp = insideTemp;
	}

	public void setOutsideTemp(float outsideTemp) {
		this.outsideTemp = outsideTemp;
	}

	public void setInsideHum(float insideHum) {
		this.insideHum = insideHum;
	}

	public void setOutsideHum(float outsideHum) {
		this.outsideHum = outsideHum;
	}

	public void setAction(ActionMap action) {
		this.action = action;
	}

	public void setState(ActionMap state) {
		this.state = state;
	}
}
