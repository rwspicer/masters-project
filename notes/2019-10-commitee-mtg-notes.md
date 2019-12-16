# Committee meeting notes 2019-10-10

The overall goal is to develop reduced order models from the Alaska Thermokarst Model (ATM) that can be used to improve other climate models such as the E3SM. The ATM exists to determine the evolution in landscape in the arctic due to the degradation of permafrost. Over the summer, I did research into various methods of reduced order modeling, and also worked on improving the current process for determining thermokarst initiation (Should the landscape be changing for a given year). Bob and I also had a conversation with scientists at berkeley (some of the developers of the E3SM), and they suggested using random forests to develop our reduced order models. This is where I'm currently focusing my work. At the moment I am working on training a random forest model against the process for determining thermokarst initiation. It's my hope that this will help determine which of inputs to the process (currently there are 4: extreme early winter precipitation, extreme total winter precipitation, extreme winter temperature, and extreme summer temperature), are most important to the process, and also verify a principal component analysis I did on these inputs. Once this is complete I would like to apply these methods to other steps in the ATM. 


BOB: Just to be clear on what is required, What is expected in regards to the project/ document. 

Lawlor: 2nd talk is progress report. 
    report: Basically a thesis 

BOB: Landslide is thermokarst event?

    Observed locs with extreme winter precip that stopped freezing of ground

    10-30 years thermokarst to stabilize. 



Lawolor 

    Dig in to sklearn methods

    Atmospheric CO2? 
    DEM STUFF


Project Goals

    DECIDE inputs (features):
        lat long, elev, slope, aspect, prior years temp, summer percip

    Document data sources? Processing

    Test against model results

    Nail data down, How to get it, what to test against?

    Maybe test in to future? not as part of CS project

    Constrain thesis to just random forests


BOB:
    Assume it's one "Feature" per gird scale.

LAWLOR:
    ELEVATION not constant


2nd talk:
    Discuss problems
    Discuss progress
    any course corrections rescoping.


Masters project report:
    Formated like thesis 
    intro, background, methods, etc
    25 - 35 Pages 


CHAPPELL said very little.
