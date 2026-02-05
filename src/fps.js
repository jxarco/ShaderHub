class FPSCounter
{
    constructor()
    {
        this.frame = 0;
        this.to = 0;
        this.fps = 0;
    }

    reset()
    {
        this.frame = 0;
        this.to = 0;
        this.fps = 60.0;
    }

    get()
    {
        return Math.floor( this.fps );
    }

    count( time )
    {
        this.frame++;

        if( ( time - this.to ) > 500.0 )
        {
            this.fps = 1000.0 * this.frame / ( time - this.to );
            this.frame = 0;
            this.to = time;
            return true;
        }

        return false;
    }
}

export { FPSCounter };