from pathlib import Path
import time
from helper_07132025.config import *
import os 

class GADDS_SLM:
    """
    This class read directly the slm file sent to server and convert them into GADDS command execution 

    """
    def __init__(self,slm_file_name, file_data, **kargs):
        self.slm_file_name= slm_file_name
        self.file_data= file_data 

    def get_command(self):
        #generate command file 
        slm_file_path= XRD_LOCAL_FOLDER+ self.slm_file_name
        print('slm file is saved to:'+slm_file_path)
        with open(slm_file_path, "w") as file:
            file.write(self.file_data) 
        print('--- SLM DATA is saved -----')
        command_name = "GADDS"
        base = "/level2 /thetatheta /port=1 /tcport=3 /COMMAND=@"
        params = """ "{}" """.format(slm_file_path)
        print('--- GADDS COMMAND is generated -----')
        return "{} {} {}".format(Path(DUMMY_DIR)/command_name, base, params)

    def get_return_file(self):
        #define a list of return file path 
        ls=[]
        num1= int(os.path.splitext(self.slm_file_name)[0])
        for i in range(FRAME):
            num2= int(i+1)
            gfrm= XRD_LOCAL_FOLDER+ f"MST_{num1:03d}_{num2:03d}.gfrm"
            ls.append(gfrm)
        return ls 

        
class GADDS:
    """
        This class work as an model class that store key parameters and is able to convert them into 
        GADDS command. Used to transfer command between XP and W10 machine
        
        Example command would look like this
        GADDS /level2 /thetatheta /port=1 /tcport=3 /COMMAND="{scan_slm}" "{id}" {temperature} {ramp_rate} {hold} {x:.03f} {y:.03f} {z:.03f}
    """
    def __init__(
        self,
        output_file_name,
        temperature,
        ramp_rate,
        hold,
        x,
        y,
        z,
        slm_file_name = "template/pxrd_scan",
        **kargs
        ):

        """
            store GADDS run parameter         
        """
        self.output_file_name = output_file_name
        self.temperature = float(temperature)
        self.ramp_rate = float(ramp_rate)
        self.hold = bool(hold)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.slm_file_name = slm_file_name

    def to_dict(self):
        return self.__dict__

    def get_command(self):
        command_name = "GADDS"
        base = "/level2 /thetatheta /port=1 /tcport=3 /COMMAND=@"
        params = """{} "{}" {} {} {} {:.03f} {:.03f} {:.03f}""".format(
            self.slm_file_name,
            self.get_return_file(),
            self.temperature,
            self.ramp_rate,
            self.hold,
            self.x, self.y, self.z
            )
        
        return "{} {} {}".format(Path(GADDS_DIR)/command_name, base, params)

    def get_return_file(self):
        if self.output_file_name == "" or self.output_file_name is None:
            self.output_file_name = str(time.time())
        return Path(TMP) / self.output_file_name
