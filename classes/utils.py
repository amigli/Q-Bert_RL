def bcd_to_decimal(byte1, byte2, byte3):
    """
    Converte tre byte in formato BCD in un valore decimale e lo rappresenta come uint32.
    
    :param byte1: Primo byte (
    :param byte2: Secondo byte
    :param byte3: Terzo byte 
    :return: Valore decimale corrispondente come int
    """

    def single_bcd_to_decimal(byte):
        value =  (byte >> 4) * 10 + (byte & 0x0F)
        return int(value)
    
    decimal_value = (
        single_bcd_to_decimal(byte1) * 10000 + 
        single_bcd_to_decimal(byte2) * 100 +   
        single_bcd_to_decimal(byte3)            
    )
    
    return decimal_value

