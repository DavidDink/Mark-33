
#include <dht.h>     // DHT library

#define dht_apin A0  // Analog pin to which the sensor is connected
dht DHT;
               /*Uncomment and comment*/
void toFahrenheit(int voltage)
{
  //Convert to celsius
  float celsius = voltage/10;
  //Convert to fahrenheit from celsius
  float fahrenheit = (celsius * 9/5) + 32;
  Serial.println(fahrenheit);  
}
void setup(void) 
{
  Serial.begin(9600);
  
     
}

void loop(void)
{
    // Read apin on DHT11
    DHT.read11(dht_apin);
    toFahrenheit((analogRead(DHT.temperature)));
    delay(500);
}
                           /*END OF FILE*/
