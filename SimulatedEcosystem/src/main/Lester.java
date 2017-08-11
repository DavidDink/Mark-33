package main;

public class Lester {
	private Session session;
	
	public Lester(Session session) {
		this.session = session;
	}
	
	public ActionMap makeDecision() {		
		Container c = session.getContainer();
		final float dTemp = controlEnv(c.getInsideFeelTemp(),
				ComfortManager.IDEAL_TEMP,
				ComfortManager.isComfortableTemp(c.getInsideFeelTemp()));
		final float dHumidity = controlEnv(c.insideHumidity(),
				ComfortManager.IDEAL_HUMIDITY,
				ComfortManager.isComfortableHumidity(c.insideHumidity()));
		return new ActionMap(dTemp, dHumidity);
	}
	
	/**
	 * Control temperature or humidity.
	 * @param currVal the current value (temp, humidity)
	 * @param desiredVal the desired value (ideal temp, humidity)
	 * @param currValInComfortZone whether or not the value (temp, humidity) is
	 * in its comfort zone
	 * @return the change in value (temp, humidity)
	 */
	public float controlEnv(float currVal, float desiredVal, 
			boolean currValInComfortZone) {
		if (!currValInComfortZone) {
			// If the value is below the desired value
			final boolean tooLow = currVal < desiredVal;
			// Measure the difference between currVal and desiredVal
			return Math.abs(desiredVal - currVal) * (tooLow ? 1f : -1f);
		} else {
			// No change needed
			return 0f;
		}
	}

	public Session getSession() {
		return session;
	}
	
	public void setSession(Session session) {
		this.session = session;
	}
}
