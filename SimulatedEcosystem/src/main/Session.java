package main;

import java.util.Random;

import main.ActionMap.Types;

public class Session {	
	private HistoryManager hisMan;
	private Container container;
	private TempChanger tempChanger;
		
	private int time;
	
	private Random random;
	
	public Session() {
		random = new Random();
	}
	
	public Session(Session other) {
		container = new Container(other.container);
		tempChanger = new TempChanger(other.tempChanger);
		time = other.time;
		// HisotryManager is not copied
	}
	
	public HistoryManager run(int numTimes) {
		clear();
		final float desiredTemp = ComfortManager.IDEAL_TEMP;
		container.setOutsideTemp(130);
		while (time < numTimes) {
			// Make time go forward
			++time;
			
			// The action that will (potentially) be taken during this iteration
			ActionMap actionMap;
			// Change in temp
			float dTemp = 0f;
			// If the temperature is in the comfort zone
			final boolean tempInComfortZone =
					ComfortManager.isComfortableTemp(container.insideTemp());
			
			final float prevInsideTemp = container.insideTemp();
			
			// If the outside temperature is colder than the inside temp,
			// or vice versa, it will have an effect on the inside temperature
			// of the container. Let's factor this in.
			final float outsideTempEffect = calculateOutsideTempEffect(
					container.insideTemp(), container.getOutsideTemp());

			// Factor in outside temp effect
			container.setInsideTemp(container.insideTemp() + outsideTempEffect);
			
			if (!tempInComfortZone) {
				// If the inside of the container is colder than the desired temp
				final boolean tooCold = container.insideTemp() < desiredTemp;
				// Measure the discrepancy between inside temp and desired temp
				final float tempDiff =
						Math.abs(desiredTemp - container.insideTemp()) * (tooCold ? 1f : -1f);
				// Get whether the AC is currently on
				final boolean acOnPreviously = tempChanger.getPower() < 0f;
				// Update the temp changer to the newly calculated power
				tempChanger.setPower(tempDiff);
				final boolean acOnNow = tempChanger.getPower() < 0f;
				ActionMap.Types state = tempDiff > 0f ? Types.HEATER_ON : Types.AC_ON;
				ActionMap.Types action = !acOnPreviously && acOnNow ? Types.AC_ON : Types.NONE;
				final float powerUsed = Math.abs(tempChanger.getPower());
				actionMap = new ActionMap(action, state, powerUsed);
				dTemp += tempChanger.getPower();
			} else {
				// Empty action
				actionMap = new ActionMap(ActionMap.Types.NONE, ActionMap.Types.NONE, 0f);
				tempChanger.setPower(0);
			}
			
			// Apply an error calculation to simulate real-life errors
			dTemp += generateError();
			
			// Update the temperature
			container.setInsideTemp(container.insideTemp() + dTemp);
			
			// If the current inside temp is the same as the one before,
			// the cooler/heater is not strong enough to compete with the outside
			// temp/environment
			if (container.insideTemp() == prevInsideTemp && !tempInComfortZone) {
				System.err.println("TempChanger is not strong enough to compete with " +
						"outside environment.");
				System.exit(0);
			}
			
//			// Add this to the history manager
			HistoryManager.Entry entry =
					new HistoryManager.Entry(new Session(this), actionMap);
			hisMan.addEntry(entry);
		}
		return hisMan;
	}
	
	/**
	 * How to calculate how much effect the outside temperature has on the inside
	 * temperature of a container.
	 */
	private float calculateOutsideTempEffect(float inside, float outside) {
		float diff = outside - inside;
		diff /= 35f;
		// Cap diff at 5
		diff = diff > 5f ? 5f : diff < -5f ? -5f : diff;
		return diff;
	}
	
	private float generateError() {
		return (2 * random.nextFloat() - 1)/10f;
	}
	
	public void clear() {
		container = new Container();
		tempChanger = new TempChanger();
		hisMan = new HistoryManager();
		time = 0;
	}
	
	public Container getContainer() {
		return container;
	}
	
	public int getCurrentTime() {
		return time;
	}
	
	public HistoryManager getHistoryManager() {
		return hisMan;
	}
}
