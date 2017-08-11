package main;

import java.util.Random;

import main.HistoryManager.Entry;

public class Session {	
	private HistoryManager hisMan;
	private Container container;
	private EnvAdjuster tempChanger;
	private EnvAdjuster humChanger;
	private Random random;
	
	public Session() {
		random = new Random();
	}
	
	public Session(Session other) {
		container = new Container(other.container);
		tempChanger = new EnvAdjuster(other.tempChanger);
		humChanger = new EnvAdjuster(other.humChanger);
		// HisotryManager is not copied
	}
	
	public HistoryManager run(int numTimes) {
		clear();
		Lester lester = new Lester(this);
		
		// The action taken
		ActionMap prevActions = new ActionMap();
		ActionMap actions = new ActionMap();

		for (int time = 1; time <= numTimes; time++) {
			// Update record of previous actions
			prevActions.set(actions);
			
			// Update the temperature
			container.adjustInsideTemp(tempChanger.getPower());
			// Update the humidity
			container.adjustInsideHumidity(humChanger.getPower());
			
			// If the outside temperature is colder than the inside temp,
			// or vice versa, it will have an effect on the inside temperature
			// of the container. Let's factor this in.
			final float outsideTempEffect = calculateEnvEffect(
					container.insideTemp(), container.outsideTemp(), 35f);
			// Same as above, but for humidity
			final float outsideHumEffect = calculateEnvEffect(
					container.insideHumidity(), container.outsideHumidity(), 40f);

			// Factor in outside temp effect
			container.adjustInsideTemp(outsideTempEffect);
			// Factor in outside humidity effect
			container.adjustInsideHumidity(outsideHumEffect);
			
			// Add error to temperature and humidity reading
			container.adjustInsideTemp(generateError());
			container.adjustInsideHumidity(generateError());
			
			// Ask engine for next action
			actions.set(lester.makeDecision());
			// Apply this action
			tempChanger.setPower(actions.getTemperatureAction());
			humChanger.setPower(actions.getHumidityAction());
			// TODO: make this more concise (constraints)
			actions.setTemperatureAction(tempChanger.getPower());
			actions.setHumidityAction(humChanger.getPower());
			
			// Add this to the history manager
			HistoryManager.Entry entry = new Entry(time, new Session(this), 
					// State = previous actions
					new ActionMap(prevActions), new ActionMap(actions));
			hisMan.addEntry(entry);
		}
		return hisMan;
	}
	
	/**
	 * Calculate how much effect the outside temperature has on the inside
	 * temperature of a container.
	 */
	private float calculateEnvEffect(float inside, float outside, float divisor) {
		float diff = outside - inside;
		diff /= divisor;
		// Cap diff at 5
		diff = diff > 5f ? 5f : diff < -5f ? -5f : diff;
		return diff;
	}
	
	private float generateError() {
		return (2f * random.nextFloat() - 1f)/10f;
	}
	
	public void clear() {
		container = new Container();
		tempChanger = new EnvAdjuster();
		humChanger = new EnvAdjuster();
		hisMan = new HistoryManager();
	}
	
	public Container getContainer() {
		return container;
	}
	
	public HistoryManager getHistoryManager() {
		return hisMan;
	}

	public EnvAdjuster getTempChanger() {
		return tempChanger;
	}

	public EnvAdjuster getHumChanger() {
		return humChanger;
	}
}
