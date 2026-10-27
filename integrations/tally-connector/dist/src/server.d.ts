import http from 'node:http';
export interface CreateBridgeServerOptions {
    /** Directory the config file lives in/is written to — usually the folder the program itself runs from. */
    baseDir: string;
}
export declare function createBridgeServer(options: CreateBridgeServerOptions): http.Server;
