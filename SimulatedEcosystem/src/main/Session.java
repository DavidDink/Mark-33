package main;

import java.util.ArrayList;
import java.util.List;

public class Session {	
	private HistoryManager hisMan;
	private Container container;
	private EnvAdjuster tempChanger;
	private EnvAdjuster humChanger;
	
	public Session() {
	}
	
	public Session(Session other) {
		container = new Container(other.container);
		tempChanger = new EnvAdjuster(other.tempChanger);
		humChanger = new EnvAdjuster(other.humChanger);
		// HisotryManager is not copied
	}
	
	public HistoryManager run(Engine engine, int numDays) {
		clear();
		
		// The action taken
		ActionMap prevActions = new ActionMap();
		ActionMap actions = new ActionMap();
		
		List<Day> days = generateDays(numDays);
		
		for (Day day : days) {
			for (int time = 0; time < day.getIntervalsPerDay(); time++) {
				// Get the current time interval
				Interval currInterval = day.getIntervals().get(time);
				// Update the outside environment to match the current interval
				container.setOutsideTemp(currInterval.getOutsideTemp());
				container.setOutsideHumidity(currInterval.getOutsideHum());
				
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
						container.getInsideFeelTemp(), container.outsideTemp(), 30f);
				// Same as above, but for humidity
				final float outsideHumEffect = calculateEnvEffect(
						container.insideHumidity(), container.outsideHumidity(), 36f);
	
				// Factor in outside temp effect
				container.adjustInsideTemp(outsideTempEffect);
				// Factor in outside humidity effect
				container.adjustInsideHumidity(outsideHumEffect);
				
				// Add error to temperature and humidity reading
				container.adjustInsideTemp(generateError());
				container.adjustInsideHumidity(generateError());
				
				// Ask engine for next action
				actions.set(engine.makeDecision(this));
				// Apply this action
				tempChanger.setPower(actions.getTemperatureAction());
				humChanger.setPower(actions.getHumidityAction());
				// TODO: make this more concise (constraints)
				actions.setTemperatureAction(tempChanger.getPower());
				actions.setHumidityAction(humChanger.getPower());
				
				// Add the to the history manager
				currInterval.update(container);
				currInterval.setState(new ActionMap(prevActions)); // State = previous actions
				currInterval.setAction(new ActionMap(actions));
				hisMan.addInterval(currInterval);
			}
		}
		return hisMan;
	}
	
	private List<Day> generateDays(int num) {
		List<Day> days = new ArrayList<>();
		for (int i = 0; i < num; i++) {
			days.add(Day.randomDay());
		}
		return days;
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
		return Utils.standardRandomFloat()/10f;
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
