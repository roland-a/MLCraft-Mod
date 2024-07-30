package net.mlcraft.elevationmod;

import java.io.*;
import java.nio.file.Path;

public class Helper {

    static byte[] readByteFile(Path path){
        try {
            var d = new DataInputStream(
                new BufferedInputStream(
                    new FileInputStream(
                        path.toFile()
                    )
                )
            );

            var a = new byte[d.available()/Byte.BYTES];
            for (int i = 0 ; i < a.length; i++){
                a[i] = d.readByte();
            }

            return a;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
